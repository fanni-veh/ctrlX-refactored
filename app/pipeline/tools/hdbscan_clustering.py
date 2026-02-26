from sklearn.cluster import HDBSCAN
import numpy as np
import pandas as pd
import onnx.numpy_helper as numpy_helper
import onnx.helper as helper
from onnx import TensorProto
import onnx


class HDBSCAN_Clustering():

    def __init__(self, **kwargs):
        """
        Initialize the HDBSCAN_Clustering class with arbitrary parameters.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments to set as class-level variables.
        """
        self.medoids = {}
        self.std_devs = []
        self.spread_factor_ax = None
        self.spread_factor_perp = None
        self.params = kwargs

    def get_param(self, key):
        return self.params.get(key, "Not found")

    def fit(self, X_train, y_training):
        """
        Perform HDBSCAN clustering on the training data and compute medoids.

        Parameters
        ----------
        X_train : np.ndarray
            The input training data.
        y_training : np.ndarray
            The training labels.

        Returns
        -------
        None
        """
        # Perform HDBSCAN Clustering
        self.clusterer = HDBSCAN(min_cluster_size=5, store_centers='medoid')

        # predict cluster
        X_train_copy = X_train.copy()
        X_train_copy.index = y_training.index

        X_train_copy['label'] = y_training
        self.clusterer.fit(X_train)

        # Compute medoids and retrieve correct label (not the predicted cluster from HDBSCAN)
        # 1 == good condition
        # 0 == bad condition

        for med in self.clusterer.medoids_:
            medoid_df = pd.DataFrame([med], columns=X_train.columns)
            label = X_train_copy[X_train_copy.drop('label', axis=1).eq(medoid_df.iloc[0]).all(axis=1)]['label'].values[0]  # Get label from the medoid
            self.medoids[label] = med

        # compute the avg std of each training cluster
        self.std_devs = []
        for label, medoid in self.medoids.items():  # TODO control for more than 2 clusters
            cluster_points = X_train_copy[X_train_copy['label'] == label].drop('label', axis=1)
            std_dev = np.std(cluster_points, axis=0)
            self.std_devs.append(np.mean(std_dev.values))
        self.spread = np.mean(np.mean(self.std_devs))

        # Error if we do not have 2 labels
        if len(self.medoids.keys()) != 2:
            raise ValueError('Fitting data did not result in two distinct clusters.')

        # New method
        # Retrieve medoids
        mu1 = self.medoids.get(1)  # good == 1
        mu0 = self.medoids.get(0)   # bad == 0

        # Compute ICV
        v = mu1 - mu0        # Vector between medoids
        v_norm = np.linalg.norm(v)
        v_hat = v / v_norm  # Unit vector along the direction of the line between the medoids

        # Mask for clusters
        mask_good = (y_training == 1)
        mask_bad = (y_training == 0)

        # Calculate axial distances
        d_0_ax = np.abs(np.dot(X_train - mu0, v_hat))  # DISTANCES of all points axially to the bad medoid
        s0_ax = np.std(d_0_ax[mask_bad], axis=0, ddof=1)  # Calculate axial std devs

        # Calculate perp distances for all points (doesnt matter if taken from good or bad medoid as reference)
        axial_component = np.dot(X_train - mu1, v_hat)[:, np.newaxis] * v_hat  # shape (N, D)
        d_perp = np.linalg.norm((X_train - mu1) - axial_component, axis=1)  # shape (N,)

        # Axial Spread factor scaled based on proportion of norm outside of std deviation regions
        spread_factor_ax = max(v_norm - s0_ax, 1e-9) / np.sqrt(-2.0 * np.log(0.15))  # Distance around near the bad cluster std bubble away from mu1 should score around 15%

        # Perp spread factor based around wanting a cylindrical sleeve around 95% of the the good training data
        spread_factor_perp = float(np.percentile(d_perp[mask_good], 95)) / np.sqrt(-2.0 * np.log(0.95))

        # Store in db
        self.spread_factor_ax = spread_factor_ax
        self.spread_factor_perp = spread_factor_perp

    def predict(self, X_data):
        """
        Predict the labels for the given data.

        Parameters
        ----------
        X_data : np.ndarray
            The input data for which labels are to be predicted.

        Returns
        -------
        np.ndarray
            The predicted labels for the input data.
        """
        predicted_labels = self.compute_labels_conf1(X_data)[0]

        return predicted_labels

    def get_params(self):
        """
        Get the parameters of the clusterer.

        Returns
        -------
        dict
            The parameters of the clusterer.
        """
        return self.clusterer.get_params()

    def compute_labels_conf1(self, mesh_points):

        # Transfer parameters
        alpha = self.get_param('alpha')
        std_dev_coeff = self.get_param('std_dev_coeff')  # keep around in case needed

        mu1 = self.medoids.get(1)  # good == 1
        mu0 = self.medoids.get(0)   # bad == 0
        spread_factor_ax = self.spread_factor_ax
        spread_factor_perp = self.spread_factor_perp

        # Vector between medoids
        v = mu1 - mu0
        v_norm = np.linalg.norm(v)
        v_hat = v / v_norm  # Unit vector along the direction of the line between the medoids

        # axial distance of test points from the good medoid along the ICV
        d_1_ax = np.abs(np.dot(mesh_points - mu1, v_hat))

        # Perpendicular distance of points projected to the ICV
        axial_component = np.dot(mesh_points - mu1, v_hat)[:, np.newaxis] * v_hat
        d_1_perp = np.linalg.norm((mesh_points - mu1) - axial_component, axis=1)

        # Calculate axial and perpendicular confdence independantly
        axial_confidence_1 = np.exp(-d_1_ax**2 / (2 * spread_factor_ax**2))  # Radial decay for c1 (Gaussian)
        perpendicular_confidence_1 = np.exp(- (alpha * d_1_perp)**2 / (2 * spread_factor_perp**2))  # Perpendicular decay

        P1 = axial_confidence_1 * perpendicular_confidence_1   # Confidence for c1 (combined effect)
        P0 = 1 - P1   # Confidence for c0 (combined effect)

        labels = np.round(P1).astype(int)  # changed to P1 instead of axial confidence

        return labels, {0: P0, 1: P1}

    def to_onnx(self):
        """
        This is the expanded version.

        Creates an ONNX model that exactly matches the logic of compute_degradation_score.
        """

        if self.medoids is None:
            raise ValueError("Medoids are not set. Ensure the model is fitted before exporting.")

        # Convert medoids to numpy arrays
        train_medoid_good = np.array(self.medoids.get(1), dtype=np.float32)
        train_medoid_bad = np.array(self.medoids.get(0), dtype=np.float32)

        train_avg_spread = self.spread

        # Define input/output tensors
        first_medoid = next(iter(self.medoids.values()))
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', first_medoid.shape[0]])
        Y_predlabel = helper.make_tensor_value_info('output_predlabel', TensorProto.FLOAT, ['batch'])
        Y_confidence = helper.make_tensor_value_info('output_confidence', TensorProto.FLOAT, ['batch', 2])
        Y_positionmetrics = helper.make_tensor_value_info('output_positionmetrics', TensorProto.FLOAT, ['batch', 3])

        # Create initializers
        inter_medoid_vector = train_medoid_bad - train_medoid_good
        vector_init = numpy_helper.from_array(inter_medoid_vector, name='inter_medoid_vector')
        good_medoid_init = numpy_helper.from_array(train_medoid_good, name='good_medoid')
        train_avg_spread_init = numpy_helper.from_array(np.array([train_avg_spread], dtype=np.float32), name='train_avg_spread')

        # Initializers to compute shape of 'repeats'
        start = numpy_helper.from_array(np.array([0], dtype=np.int64), name='start')

        # Compute denominator (dot product of inter_medoid_vector with itself)
        denominator = np.dot(inter_medoid_vector, inter_medoid_vector).astype(np.float32)
        denominator_init = numpy_helper.from_array(np.array([denominator], dtype=np.float32), name='denominator')

        # Constants for clipping
        min_val = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name='min_val')
        max_val = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name='max_val')
        one_tensor = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name='one_tensor')
        axes_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), name='axes_1')

        nodes = [
            # 1. Get the shape of the input
            helper.make_node(
                'Shape',
                inputs=['input'],
                outputs=['input_shape'],
                name='get_input_shape'
            ),
            # 2. Extract the batch size (first dimension) using Slice
            helper.make_node(
                'Slice',
                inputs=['input_shape', 'start', 'axes_1', 'start'],
                outputs=['batch_size'],
                name='extract_batch_size'
            ),
            # 3. Concatenate batch size with 1 to form repeats
            helper.make_node(
                'Concat',
                inputs=['batch_size', 'axes_1'],
                outputs=['repeats'],
                name='concat_repeats',
                axis=0
            ),
            # 1. Calculate vector from good medoid to point of interest
            helper.make_node(
                'Sub',
                inputs=['input', 'good_medoid'],
                outputs=['point_vectors'],
                name='subtract_good_medoid'
            ),

            # 2. Calculate numerator (dot product with inter_medoid_vector)
            helper.make_node(
                'MatMul',
                inputs=['point_vectors', 'inter_medoid_vector'],
                outputs=['numerator'],
                name='dot_product'
            ),

            # 3. Squeeze numerator to remove extra dimension
            helper.make_node(
                'Squeeze',
                inputs=['numerator'],
                outputs=['squeezed_numerator'],
                name='squeeze_numerator'
            ),

            # 4. Calculate X_projection by dividing by denominator
            helper.make_node(
                'Div',
                inputs=['squeezed_numerator', 'denominator'],
                outputs=['X_projections'],
                name='compute_X_projections'
            ),

            # 5. Clip values between 0 and 1
            helper.make_node(
                'Clip',
                inputs=['X_projections', 'min_val', 'max_val'],
                outputs=['clipped_X_projections'],
                name='clip_projections'
            ),

            # 6. Compute 1 - prob for conf1
            helper.make_node(
                'Sub',
                inputs=['one_tensor', 'clipped_X_projections'],
                outputs=['conf1'],
                name='compute_conf1'
            ),
            helper.make_node(
                'Unsqueeze',
                inputs=['conf1', 'axes_1'],
                outputs=['output_conf1'],
                name='unsqueeze_conf1'
            ),
            helper.make_node(
                'Unsqueeze',
                inputs=['clipped_X_projections', 'axes_1'],
                outputs=['output_conf0'],
                name='unsqueeze_conf0'
            ),
            helper.make_node(
                'Concat',
                inputs=['output_conf0', 'output_conf1'],
                outputs=['output_confidence'],
                name='concat_confidence',
                axis=1
            ),

            # 8. Compute predicted label by rounding 1 - clipped_X_projections
            helper.make_node(
                'Sub',
                inputs=['one_tensor', 'clipped_X_projections'],
                outputs=['output_predlabel_not_rounded'],
                name='subtract_from_one'
            ),
            helper.make_node(
                'Round',
                inputs=['output_predlabel_not_rounded'],
                outputs=['output_predlabel'],
                name='round_predlabel'
            ),

            # 9. Compute the vector from inter_medoid_axis to point of interest (orthogonal distance)
            helper.make_node(
                'Unsqueeze',
                inputs=['X_projections', 'axes_1'],
                outputs=['X_projections_unsqueezed'],
                name='unsqueeze_X_projections'
            ),
            helper.make_node(
                'Mul',
                inputs=['X_projections_unsqueezed', 'inter_medoid_vector'],
                outputs=['projected_vectors'],
                name='scale_inter_medoid'
            ),
            helper.make_node(
                'Sub',
                inputs=['point_vectors', 'projected_vectors'],
                outputs=['difference_vectors'],
                name='vector_difference'
            ),
            helper.make_node(
                'ReduceSumSquare',
                inputs=['difference_vectors'],
                outputs=['difference_squared'],
                name='compute_squared_difference',
                axes=[1]
            ),
            helper.make_node(
                'Sqrt',
                inputs=['difference_squared'],
                outputs=['vector_norm'],
                name='compute_norm'
            ),

            # 10. Compute the std of the testing data
            helper.make_node(
                'ReduceMean',
                inputs=['input'],
                outputs=['input_mean'],
                name='compute_input_mean',
                axes=[0]
            ),
            helper.make_node(
                'Sub',
                inputs=['input', 'input_mean'],
                outputs=['input_centered'],
                name='subtract_mean'
            ),
            helper.make_node(
                'Mul',
                inputs=['input_centered', 'input_centered'],
                outputs=['squared_diffs'],
                name='compute_squared_diffs'
            ),
            helper.make_node(
                'ReduceMean',
                inputs=['squared_diffs'],
                outputs=['mean_squared_diffs'],
                name='compute_mean_squared_diffs',
                axes=[0]
            ),
            helper.make_node(
                'Sqrt',
                inputs=['mean_squared_diffs'],
                outputs=['mean_squared_diffs_std'],
                name='compute_std'
            ),
            helper.make_node(
                'ReduceMean',
                inputs=['mean_squared_diffs_std'],
                outputs=['mean_std'],
                name='compute_mean_std',
                axes=[1]
            ),

            # 11. Compute SpreadRatio by dividing with training spread
            helper.make_node(
                'Div',
                inputs=['mean_std', 'train_avg_spread'],
                outputs=['spread_ratio'],
                name='divide_by_avg_spread'
            ),
            # Repeat SpreadRatio for all points using Tiles
            helper.make_node(
                'Tile',
                inputs=['spread_ratio', 'repeats'],
                outputs=['spread_ratio_tiled'],
                name='tile_spread_ratio'
            ),

            # 10. Concatenate projections and norm for position metrics
            helper.make_node(
                'Concat',
                inputs=['X_projections_unsqueezed', 'vector_norm', 'spread_ratio_tiled'],
                outputs=['output_positionmetrics'],
                name='concat_positionmetrics',
                axis=1
            ),
        ]

        # Create the graph
        graph = helper.make_graph(
            nodes=nodes,
            name='HDBSCAN_Predictor',
            inputs=[X],
            outputs=[Y_predlabel, Y_confidence, Y_positionmetrics],  # Y_confidence should be computed based on distance metrics.
            initializer=[vector_init, good_medoid_init, denominator_init, axes_1, train_avg_spread_init, min_val, max_val, one_tensor, start]
        )

        # Create and validate the model
        model = helper.make_model(
            graph,
            producer_name='CustomHDBSCAN',
            opset_imports=[helper.make_operatorsetid("", 13)]
        )

        onnx.checker.check_model(model)
        return model
