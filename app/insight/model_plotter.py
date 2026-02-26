from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import onnxruntime as ort
import os
from app.pipeline.tools.onnx_converter import sk_to_onnx
from app.pipeline.tools.hdbscan_clustering import HDBSCAN_Clustering
import plotly.graph_objects as go

import plotly.io as pio

pio.templates.default = "plotly_dark"


def plot_model_2D(db_model, targetFolder, X_train=None, y_train=None, X_validation=None, y_validation=None, quality=5):
    """
    Plots the decision boundary of a model after applying PCA and includes validation data.

    :param model: The trained model.
    :param x_train: Training data features.
    :param y_train: Training data labels.
    :param x_validation: Validation data features.
    :param y_validation: Validation data labels.
    :param targetFilename: The filename for saving the plot.
    :param targetFolder: The folder where the plot will be saved.
    :param model_metadata: Dictionary containing model metadata to display on the plot.
    """

    matplotlib.use('Agg')

    # Apply PCA to the training data
    pca = PCA(n_components=2)
    pca.mean_ = np.array(db_model.model_metadata['pca_params']['mean'])
    pca.components_ = np.array(db_model.model_metadata['pca_params']['components'])
    pca.explained_variance_ = np.array(db_model.model_metadata['pca_params']['explained_variance'])
    pca.explained_variance_ratio_ = np.array(db_model.model_metadata['pca_params']['explained_variance_ratio'])
    pca.singular_values_ = np.array(db_model.model_metadata['pca_params']['singular_values'])

    if X_train is not None:
        X_train = X_train[sorted(X_train.columns)]
        x_train_pca = pca.transform(X_train)

    if X_validation is not None:
        X_validation = X_validation[sorted(X_validation.columns)]
        x_validation_pca = pca.transform(X_validation)

    # Create a custom colormap from red to blue
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
    cmap_bold = ListedColormap(['#0000FF', '#FF0000'])

    # Plotting the decision boundary in 2D
    plt.figure(figsize=(10, 8))

    x_min, x_max, y_min, y_max = -3, 3, -3, 3

    # Define the range for the meshgrid based on available data
    if X_train is not None and X_validation is not None:
        x_min = min(x_train_pca[:, 0].min(), x_validation_pca[:, 0].min()) * 1.1
        x_max = max(x_train_pca[:, 0].max(), x_validation_pca[:, 0].max()) * 1.1
        y_min = min(x_train_pca[:, 1].min(), x_validation_pca[:, 1].min()) * 1.1
        y_max = max(x_train_pca[:, 1].max(), x_validation_pca[:, 1].max()) * 1.1
    elif X_train is not None:
        x_min = min(x_min, x_train_pca[:, 0].min()) * 1.1
        x_max = max(x_max, x_train_pca[:, 0].max()) * 1.1
        y_min = min(y_min, x_train_pca[:, 1].min()) * 1.1
        y_max = max(y_max, x_train_pca[:, 1].max()) * 1.1
    elif X_validation is not None:
        x_min = min(x_min, x_validation_pca[:, 0].min()) * 1.1
        x_max = max(x_max, x_validation_pca[:, 0].max()) * 1.1
        y_min = min(y_min, x_validation_pca[:, 1].min()) * 1.1
        y_max = max(y_max, x_validation_pca[:, 1].max()) * 1.1

    # Define the base for exponential scaling and the maximum points at quality 10
    base = 1.2  # rate of growth
    max_points = 100  # Maximum points at quality 10

    # Calculate the number of points using exponential scaling
    n_points = int(base ** (quality - 1) * max_points)
    if n_points < 5:
        n_points = 5

    # Calculate the step size h based on the number of points
    h_x = (x_max - x_min) / n_points
    h_y = (y_max - y_min) / n_points

    # Create a mesh to plot decision boundary
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Transforms the 2D coords to be plotted, into the corresponding point in feature space to use in prediction.
    pred_input = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

    if 'HDBSCAN_Clustering' in db_model.model_name:
        # alpha = db_model.model_metadata['model_parameters']['alpha']
        clustering = HDBSCAN_Clustering(**db_model.model_metadata['params'])
        clustering.medoids = {int(float(k)): np.array(v) for k, v in db_model.model_metadata['medoids'].items()}
        clustering.std_devs = np.array(db_model.model_metadata['std_devs'], dtype='float')
        clustering.spread_factor_ax = db_model.model_metadata['spread_factor_ax']
        clustering.spread_factor_perp = db_model.model_metadata['spread_factor_perp']
        _, Z = clustering.compute_labels_conf1(pred_input)
        Z = Z[1]  # extract conf_1 values

    else:
        if type(db_model.model_onnx) is not bytes:
            input_dim = db_model.model_metadata['input_dim']
            db_model.model_onnx = sk_to_onnx(input_dim, db_model.model_onnx)

        session = ort.InferenceSession(db_model.model_onnx)

        # Use the session to make predictions
        # Extract input names
        input_name = session.get_inputs()[0].name

        Z = session.run(None, {input_name: pred_input.astype(np.float32)})[1]
        Z = np.array([d[1] for d in Z])  # extract conf_1 values

    # Reshape the N(Z) conf1 values into the plotting mesh shape
    Z = Z.reshape(xx.shape)

    contour = plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.bwr_r)

    # Plotting the training data if provided
    if X_train is not None:
        train_colors = np.where(y_train == 0, 'red', 'blue')
        plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=train_colors, edgecolors='k', label='Training Data')

    # Plotting the validation data if provided
    if X_validation is not None:
        validation_colors = np.where(y_validation == 0, 'red', 'blue')
        plt.scatter(x_validation_pca[:, 0], x_validation_pca[:, 1], c=validation_colors, marker='x', label='Validation Data')

    plt.title(f'{db_model.model_name}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    # Adding colorbar for the contour plot
    cbar = plt.colorbar(contour)
    cbar.set_label('Probability of Good')

    # Adding legend for clarity
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color='w', marker='o', markerfacecolor='red', markersize=10, linestyle='None', label='Bad (Training Data)'),
            plt.Line2D([0], [0], color='w', marker='o', markerfacecolor='blue', markersize=10, linestyle='None', label='Good (Training Data)'),
            plt.Line2D([0], [0], color='red', marker='x', markerfacecolor='red', markersize=10, linestyle='None', label='Bad (Validation Data)'),
            plt.Line2D([0], [0], color='blue', marker='x', markerfacecolor='blue', markersize=10, linestyle='None', label='Good (Validation Data)')
        ],
        loc='best'
    )

    # Save the plot to the specified folder and filename
    plt.savefig(os.path.join(targetFolder, db_model.model_name + '.png'), bbox_inches='tight')
    plt.close()


def plot_model_3D(db_model,
                  targetFolder,
                  X_train=None,
                  y_train=None,
                  X_validation=None,
                  y_validation=None,
                  quality=5,
                  output_file="3d_plot.html",
                  show_vectors=False,
                  show_model=True,
                  show_data=True):
    """
    Plots an interactive 3D scatter plot from a pandas DataFrame and saves it as an HTML file.

    Parameters:
        df (pd.DataFrame): A DataFrame with exactly three numerical columns.
        output_file (str): The filename to save the HTML plot.

    Returns:
        None
    """

    # Apply PCA to the training data
    # pca = PCA(n_components=2)
    # pca.mean_ = np.array(db_model.model_metadata['pca_params']['mean'])
    # pca.components_ = np.array(db_model.model_metadata['pca_params']['components'])
    # pca.explained_variance_ = np.array(db_model.model_metadata['pca_params']['explained_variance'])
    # pca.explained_variance_ratio_ = np.array(db_model.model_metadata['pca_params']['explained_variance_ratio'])
    # pca.singular_values_ = np.array(db_model.model_metadata['pca_params']['singular_values'])

    # pc1, pc2 = pca.components_[:2]

    # Transforms the 2D coords to be plotted, into the corresponding point in feature space to use in prediction.
    # pred_input = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

    # Extract column names
    x_col, y_col, z_col = X_validation.columns

    # Determine axis limits
    x_min, x_max = min(X_validation[x_col].min(), -3), max(X_validation[x_col].max(), 3)
    y_min, y_max = min(X_validation[y_col].min(), -3), max(X_validation[y_col].max(), 3)
    z_min, z_max = min(X_validation[z_col].min(), -3), max(X_validation[z_col].max(), 3)

    # Create an empty figure
    fig = go.Figure()

    # Create a mesh to plot decision boundary
    x_vals = np.linspace(x_min, x_max, quality)
    y_vals = np.linspace(y_min, y_max, quality)
    z_vals = np.linspace(z_min, z_max, quality)
    xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    coords = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    if db_model is not None:

        if 'HDBSCAN_Clustering' in db_model.model_name:
            # alpha = db_model.model_metadata['model_parameters']['alpha']
            clustering = HDBSCAN_Clustering(**db_model.model_metadata['params'])
            clustering.medoids = {int(float(k)): np.array(v) for k, v in db_model.model_metadata['medoids'].items()}
            clustering.std_devs = np.array(db_model.model_metadata['std_devs'], dtype='float')
            clustering.spread_factor_ax = db_model.model_metadata['spread_factor_ax']
            clustering.spread_factor_perp = db_model.model_metadata['spread_factor_perp']
            _, Z = clustering.compute_labels_conf1(coords)
            conf_1 = Z[1]  # extract conf_1 values
            conf_1 = np.round(conf_1, 2)
            isomin = 0.05
            isomax = 0.85
            surface_count = 5

            # Add vectors showing the steps in HDBSCAN
            if show_vectors:

                medoid0 = clustering.medoids[0]
                medoid1 = clustering.medoids[1]
                std_dev0 = clustering.std_devs[0]
                std_dev1 = clustering.std_devs[1]

                # Medoids
                fig.add_trace(go.Scatter3d(
                    x=[medoid0[0], medoid1[0]],
                    y=[medoid0[1], medoid1[1]],
                    z=[medoid0[2], medoid1[2]],
                    mode='markers',
                    marker=dict(size=10, color='black', symbol='x'),
                    name="Medoids"
                ))

                # Std_dev
                for medoid_center, std_dev, color in zip([medoid0, medoid1], [std_dev0, std_dev1], ['red', 'blue']):

                    # Create a sphere around the medoid center with radius std_dev
                    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
                    x_sphere = medoid_center[0] + std_dev * np.cos(u) * np.sin(v)
                    y_sphere = medoid_center[1] + std_dev * np.sin(u) * np.sin(v)
                    z_sphere = medoid_center[2] + std_dev * np.cos(v)
                    fig.add_trace(go.Surface(
                        x=x_sphere, y=y_sphere, z=z_sphere,
                        opacity=0.5,
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        name=f"Medoid Sphere ({color})"
                    ))

                # Inter cluster vector
                fig.add_trace(go.Scatter3d(
                    x=[medoid0[0], medoid1[0]],
                    y=[medoid0[1], medoid1[1]],
                    z=[medoid0[2], medoid1[2]],
                    mode='lines',
                    line=dict(color='green', width=10),
                    showlegend=False,
                    opacity=1
                ))

                # Free Length Vector
                direction = medoid1 - medoid0  # Direction vector from medoid0 to medoid1 and normalize
                direction_norm = direction / np.linalg.norm(direction)
                point0 = medoid0 + direction_norm * std_dev0  # Point std_dev0 away from medoid0 toward medoid1
                point1 = medoid1 - direction_norm * std_dev1  # Point std_dev1 away from medoid1 toward medoid0 (opposite direction)
                inter_cluster_free_length = np.linalg.norm(point1 - point0) / np.linalg.norm(direction)

                fig.add_trace(go.Scatter3d(
                    x=[point0[0], point1[0]],
                    y=[point0[1], point1[1]],
                    z=[point0[2], point1[2]],
                    mode='lines+text',
                    line=dict(color='green', width=30),
                    name=f"Inter cluster Free Length",
                    text=["", f"Inter cluster Free Length (len/norm={inter_cluster_free_length:.2f})"],
                    textposition="middle right",
                    showlegend=True,
                    opacity=1
                ))

                # Text with additional data
                ratio = std_dev1 / std_dev0

                # Add annotation fixed to the bottom of the screen (outside 3D scene)
                fig.add_annotation(
                    text=f"std_dev1/std_dev0 ratio: {ratio:.2f}<br>Inter cluster Free Length: {inter_cluster_free_length:.2f}",
                    xref="paper", yref="paper",
                    x=0.01, y=0.01,  # Bottom left
                    showarrow=False,
                    font=dict(size=30, color='black'),
                    align="left",
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="green",
                    borderwidth=1,
                )

        else:  # If the model is not HDBSCAN, use ONNX for predictions

            if type(db_model.model_onnx) is not bytes:
                input_dim = db_model.model_metadata['input_dim']
                db_model.model_onnx = sk_to_onnx(input_dim, db_model.model_onnx)

            session = ort.InferenceSession(db_model.model_onnx)
            input_name = session.get_inputs()[0].name
            Z = session.run(None, {input_name: coords.astype(np.float32)})[1]
            conf_1 = np.array([d[1] for d in Z])  # extract conf_1 values
            conf_1 = np.round(conf_1, 2)
            conf_1 = conf_1.reshape(xx.shape)
            isomin = 0.05
            isomax = 0.95
            surface_count = 3

            # Plot the points that form the isosurface (within the isomin/isomax range)
            mask = (conf_1 >= isomin) & (conf_1 <= isomax)
            mask_flat = mask.ravel()
            fig.add_trace(go.Scatter3d(
                x=coords[mask_flat, 0],
                y=coords[mask_flat, 1],
                z=coords[mask_flat, 2],
                mode='markers',
                marker=dict(size=3, color=conf_1.ravel()[mask_flat], colorscale="RdBu", opacity=0.7),
                name="Model evaluation points"
            ))

        if show_model:
            # Add 3D Contour (Isosurface)
            fig.add_trace(go.Isosurface(
                x=coords[:, 0],  # X-coordinates
                y=coords[:, 1],  # Y-coordinates
                z=coords[:, 2],  # Z-coordinates
                value=conf_1,
                isomin=isomin,
                isomax=isomax,  # Adjust contour levels
                opacity=0.3,
                surface_count=surface_count,  # Number of contour levels
                colorscale="RdBu",
                showscale=False
            ))

    if X_train is not None and show_data:

        train_colors = ['red' if y == 0 else 'blue' for y in y_train]
        fig.add_trace(go.Scatter3d(
            x=X_train[x_col], y=X_train[y_col], z=X_train[z_col],
            mode='markers',
            marker=dict(color=train_colors, size=5, opacity=0.8),
            name="Training Data"
        ))

        # Transform both Training & Validation Data into PCA space
        # X_train_pca = pca.transform(X_train)  # 2D representation
        # X_train_proj = pca.inverse_transform(X_train_pca)

        # # Add projected training data
        # fig.add_trace(go.Scatter3d(
        #     x=X_train_proj[:, 0], y=X_train_proj[:, 1], z=X_train_proj[:, 2],
        #     mode='markers',
        #     marker=dict(symbol="circle", color=train_colors, size=4, opacity=0.6),
        #     name="Train Projection"
        # ))

    if X_validation is not None and show_data:

        validation_colors = ['red' if y == 0 else 'blue' for y in y_validation]
        fig.add_trace(go.Scatter3d(
            x=X_validation[x_col], y=X_validation[y_col], z=X_validation[z_col],
            mode='markers',
            marker=dict(symbol="x", color=validation_colors, size=6, opacity=0.8),
            name="Validation Data"
        ))

        # Convert PCA points back to 3D (for visualization)
        # X_validation_pca = pca.transform(X_validation)  # 2D representation
        # X_validation_proj = pca.inverse_transform(X_validation_pca)

        # # Add projected validation data
        # fig.add_trace(go.Scatter3d(
        #     x=X_validation_proj[:, 0], y=X_validation_proj[:, 1], z=X_validation_proj[:, 2],
        #     mode='markers',
        #     marker=dict(symbol="circle", color=validation_colors, size=4, opacity=0.6),
        #     name="Validation Projection"
        # ))

    # Update layout
    fig.update_layout(title="Training and Validation Data",
                      scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col))

    # Generate the PCA plane
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    # zz = pca.mean_[2] + (pc1[2] * (xx - pca.mean_[0]) + pc2[2] * (yy - pca.mean_[1])) / pc1[0]
    # fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.5, colorscale='Viridis', showscale=False))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            zaxis=dict(range=[z_min, z_max])
        )
    )

    # Save to an interactive HTML file
    if db_model is None:
        fig_name = '3D_data_plot.html'
    else:
        if show_vectors:
            fig_name = db_model.model_name + '_vector.html'
        else:
            fig_name = db_model.model_name + '.html'
    fig.write_html(os.path.join(targetFolder, fig_name))
