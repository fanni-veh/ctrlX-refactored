"""
The select_feature_auto.py file currently utilizes a range of dimensionality reduction techniques prior to being directly applied to the ML model.
"""
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from app.scripts.logging_summaries import dataframe_metadata, plot_correlation_histogram, correlated_features_summary
from app.scripts.tsk_param import TskParam
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from app.config import setting
from kneed import KneeLocator
import heapq


def run_pca_keep_pca(tsk_param: TskParam, train_only_features_ft_df, validation_only_features_ft_df):
    """
    This function performs PCA:
        - Extracts the number of principal components needed to achieve a desired level of explained variance
        - extracts the most important features contributing to each principal component
        - returns a dataframe containing the reduced feature set

    :param features: feature set containing normalised feature values for cycles in the training set

    :return: extracted_columns_df - reduced feature set after PCA dimensionality reduction
    """
    # If desired_variance_retained < 1, retain % of the variance of the original data.
    # E.g. desired_variance_retained = 0.5 -> keep 50% of the variance of the data
    # If desired_variance_retained >= 1, this specifies the actual number of principal components to extract
    desired_variance_retained = 2

    # PCA initialisation - obtain principal components that provide the retention of variance in original data
    pca = PCA(n_components=desired_variance_retained)

    # computes the PCs, transform features, pca_result = (num training cycles x pca.n_components_)
    pca.fit(train_only_features_ft_df)

    train_pca_result = pca.transform(train_only_features_ft_df)
    validation_pca_result = pca.transform(validation_only_features_ft_df)

    # Extract the relevant columns from the original DataFrame
    columns = [f'PC{i+1}' for i in range(desired_variance_retained)]
    train_pca_df = pd.DataFrame(train_pca_result, columns=columns)
    train_pca_df.index = train_only_features_ft_df.index
    validation_pca_df = pd.DataFrame(validation_pca_result, columns=columns)
    validation_pca_df.index = validation_only_features_ft_df.index

    tsk_param.logger.debug('Finished PCA')
    tsk_param.logger.debug(dataframe_metadata(train_pca_df, df_name='Train PCA output'))
    tsk_param.logger.debug(dataframe_metadata(validation_pca_df, df_name='Test PCA output'))

    return train_pca_df, validation_pca_df


def run_pca(tsk_param: TskParam, features_ft_df, sequence_order):
    """
    This function performs PCA:
        - Extracts the number of principal components needed to achieve a desired level of explained variance
        - extracts the most important features contributing to each principal component
        - returns a dataframe containing the reduced feature set

    :param features: feature set containing normalised feature values for cycles in the training set
    :param sequence_order: sequence order of DR method - -1 = last method, -2 = second last, -3 = third last.

    :return: extracted_columns_df - reduced feature set after PCA dimensionality reduction
    """
    # If desired_variance_retained < 1, retain % of the variance of the original data.
    # E.g. desired_variance_retained = 0.5 -> keep 50% of the variance of the data
    # If desired_variance_retained >= 1, this specifies the actual number of principal components to extract
    # desired_variance_retained = 1

    pca = PCA(n_components=1)
    n_features_important = setting.pca_feature_output_by_order[sequence_order]

    # computes the PCs, transform features, pca_result = (num training cycles x pca.n_components_)
    pca.fit_transform(features_ft_df)

    # number of principal components
    num_pcs = pca.n_components_

    # Initialize an empty set to store the important columns
    important_features_set = set()

    # For each relevant principal component (total given by num_components), extract the top n_features_important features that contributed to that principal component the most
    for i in range(num_pcs):
        # Take the top n_features_important features from the most relevant PC, and store it in set.
        important_features_this_PC = features_ft_df.columns[abs(pca.components_[i]).argsort()[::-1][:n_features_important]]
        important_features_set = important_features_set.union(important_features_this_PC)

    # Extract the relevant columns from the original DataFrame
    extracted_columns_ft_df = features_ft_df[list(important_features_set)]

    tsk_param.logger.debug('Finished PCA')
    tsk_param.logger.debug(dataframe_metadata(extracted_columns_ft_df, df_name='PCA output'))

    return extracted_columns_ft_df


def run_correlation_matrix(tsk_param: TskParam, extracted_features_ft_df, sequence_order, threshold=0.75, method='pearson'):
    """
    Eliminates features with high correlation
        - Computes the correlation matrix for the input feature df.
        - Calculate number of features retained per 0.01 <= corr thresholds < 1
        - Determine threshold required to output input number of features
        - Remove high correlation features.
        - Returns dataframe of training cycles with reduced number of features based on correlation.

    :param extracted_features_ft_df: normalised features df containing num training cycles x total features without cycle_id or label
    :param sequence_order: sequence order of DR method - -1 = last method, -2 = second last, -3 = third last.
    :return: dataframe containing all training cycles against a reduced number of features based on correlation.

    """

    # Compute the correlation matrix - size is num features x num features
    corr_matrix = extracted_features_ft_df.corr(method=method).abs()

    # get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype('bool'))

    # Create a df containing thresholds vs the number of features retained
    corr_thresholds = np.arange(0.01, 1.0, 0.01)
    comparison = upper.values[:, :, None] >= corr_thresholds
    count_above_threshold = comparison.any(axis=0).sum(axis=0)
    fts_retained = len(upper.columns) - count_above_threshold   # Calculate the number of features retained at 0.01 <= thresholds < 1
    df_threshold_ft_counts = pd.DataFrame({'threshold': corr_thresholds, 'fts_retained': fts_retained})

    # Look up the threshold required to output the desired number of features post correlation as per param.
    threshold = df_threshold_ft_counts.loc[df_threshold_ft_counts['fts_retained'] <= setting.corr_feature_output_by_order[sequence_order], 'threshold'].max()

    # Drop highly correlated features
    columns_to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]  # identify features to remove
    selected_features_ft_df = extracted_features_ft_df.drop(columns_to_drop, axis=1)

    tsk_param.logger.debug('Finished correlation filtering on %d threshold.', threshold)
    tsk_param.logger.debug(dataframe_metadata(selected_features_ft_df, df_name='Correlation filtering output'))

    return selected_features_ft_df, upper


def calculate_feature_correlations(tsk_param: TskParam, extracted_features_ft_df):
    """
    Calculates the Point-Biserial Correlation between each feature and the data labels.

    This function computes the correlation between each feature and the binary label using the Point-Biserial Correlation coefficient.
    The results are stored in a DataFrame with the feature names as column names.

    The Point-Biserial Correlation Coefficient (rₚ) measures the strength and direction of the association between a binary variable
    and a continuous variable. It is a special case of the Pearson correlation coefficient where one variable is binary.

    Mathematically, the Point-Biserial Correlation Coefficient is given by:

    \[
    r_p = \frac{M_1 - M_0}{s} \cdot \sqrt{\frac{n_1 \cdot n_0}{n^2}}
    \]

    where:
    - \(M_1\) is the mean of the continuous variable for the group where the binary variable is 1.
    - \(M_0\) is the mean of the continuous variable for the group where the binary variable is 0.
    - \(s\) is the standard deviation of the continuous variable.
    - \(n_1\) is the number of observations where the binary variable is 1.
    - \(n_0\) is the number of observations where the binary variable is 0.
    - \(n\) is the total number of observations.

    The coefficient ranges from -1 to 1:
    - A value of 1 indicates a perfect positive correlation.
    - A value of -1 indicates a perfect negative correlation.
    - A value of 0 indicates no correlation.


    Parameters:
        tsk_param (TskParam): An object containing parameters and configuration, including a logger and output path.
        extracted_features_ft_df (pd.DataFrame): DataFrame containing features and a binary label column named 'label'.

    Returns:
        pd.DataFrame: A DataFrame with a single row where each column represents a feature and its value is the Point-Biserial Correlation
                      between that feature and the label.

    """

    # Separate features and label
    features_only_ft_df = extracted_features_ft_df.drop(columns=["label"])
    labels = extracted_features_ft_df["label"]

    # Initialize lists to store results
    correlation_results = {}

    # Calculate Point-Biserial Correlation for each feature
    for column in features_only_ft_df.columns:
        corr, p_value = pointbiserialr(labels, features_only_ft_df[column])
        correlation_results[column] = corr

    # Convert results into a DataFrame
    results_df = pd.DataFrame([correlation_results], columns=features_only_ft_df.columns)

    tsk_param.logger.debug('Finished correlation analysis wrt labels.')
    tsk_param.logger.debug(correlated_features_summary(results_df))

    if setting.plot_corr_labels:
        plot_correlation_histogram(results_df, 'correlation_to_labels', tsk_param.output_path)


def run_advanced_correlation_matrix(tsk_param: TskParam, extracted_features_ft_df, threshold=0.75, method='pearson'):
    """
    Eliminates features with high correlation
        - Computes the correlation matrix for the input feature df.
        - Remove high correlation features.
        - Returns dataframe of training cycles with reduced number of features based on correlation.

    :param extracted_features_ft_df: normalised features df containing num training cycles x total features without cycle_id or label
    :return: dataframe containing all training cycles against a reduced number of features based on correlation.

    """

    # Compute the correlation matrix - size is num features x num features
    corr_matrix = extracted_features_ft_df.corr(method=method).abs()

    # get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype('bool'))

    # Drop highly correlated features
    columns_to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]  # identify features to remove
    selected_features_ft_df = extracted_features_ft_df.drop(columns_to_drop, axis=1)

    tsk_param.logger.debug('Finished correlation filtering on %d threshold.', threshold)
    tsk_param.logger.debug(dataframe_metadata(selected_features_ft_df, df_name='Correlation filtering output'))

    return selected_features_ft_df, upper


def run_svd(tsk_param: TskParam, input_features_ft_df, sequence_order):
    """
    Does a Truncated Singular Value Decomposition (SVD) on the input feature DataFrame to reduce its dimensionality,
    selecting features based on their importance across the components.

    The process:
    1. Applying SVD to the input DataFrame `input_features_ft_df` to decompose it into three matrices (U, S, V*).
    2. Calculating the cumulative explained variance to determine the optimal number of components that exceed a predefined threshold.
    3. Extracting the most important features that contribute to each selected component.
    4. Returning a DataFrame containing the columns corresponding to the important features.

    Parameters:
        :param input_features_ft_df (pd.DataFrame): The input DataFrame containing the features to be reduced.
        :param sequence_order: sequence order of DR method - -1 = last method, -2 = second last, -3 = third last.

    Returns:
        pd.DataFrame: A DataFrame containing the selected important features after dimensionality reduction.
    """

    num_components = 1
    svd = TruncatedSVD(n_components=num_components)  # Adjust the maximum number of components if needed

    # result is the matrix (U) in the SVD factorisation (A = USV*) of size (num cycles x output_components)
    svd.fit_transform(input_features_ft_df)

    # Get the singular values - These are the diagonal values of the diagonal matrix (S) in the SVD factorisation (A = USV*)
    # singular_values = svd.singular_values_

    # Determine the index where the cumulative explained variance ratio exceeds the threshold
    # desired_cumulative_threshold = setting.svd_threshold
    # cumulative_explained_variance = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    # # is svd10.singular_values_**2/np.sum(svd10.singular_values_) the same as svd.explained_variance just without normalisation?
    # num_components = np.argmax(cumulative_explained_variance >= desired_cumulative_threshold) + 1

    # Feature importance extraction
    n_features_important = setting.svd_feature_output_by_order[sequence_order]
    important_features_set = set()

    # Loop through the num_components to select the top 'n_features_important' features contributing to this component
    for i in range(num_components):
        # Select the top 'n_features_important' features.
        important_features = input_features_ft_df.columns[abs(svd.components_[i]).argsort()[::-1][:n_features_important]]
        important_features_set = important_features_set.union(important_features)

    # Extract the relevant columns from the original DataFrame
    extracted_columns_df = input_features_ft_df[list(important_features_set)]

    tsk_param.logger.debug('Finished SVD - Output %d features.', n_features_important)
    # tsk_param.logger.debug('Finished SVD using %d variance threshold.', desired_cumulative_threshold)
    tsk_param.logger.debug(dataframe_metadata(extracted_columns_df, df_name='SVD output'))

    return extracted_columns_df


def run_overlap(tsk_param: TskParam, input_features_ft_df, threshold=0.2, n_dim_output=3):
    """
    Run the overlap method for dimensionality reduction.
    Part of the feature selection process.

    :param input_features_ft_df: feature set containing normalised feature values for cycles in the training set
    :return: extracted_columns_df - reduced feature set after overlap dimensionality reduction.
    """

    # Separate bad and good signals, drop label and cycle_id not required in feature selection
    class_0_data = input_features_ft_df[input_features_ft_df['label'] == 0].drop(columns=['label'])
    class_1_data = input_features_ft_df[input_features_ft_df['label'] == 1].drop(columns=['label'])

    # calculate the number of histogram bins used based on the total number of cycles in the data.
    total_num_cycles = len(class_0_data) + len(class_1_data)
    num_bins = round(2 * total_num_cycles ** (1.0 / 3))

    roi_dict = {}
    # For each feature in the dataset, calculate the ROI value
    for feature in class_0_data.columns:
        nioDistrib = class_0_data[feature]
        ioDistrib = class_1_data[feature]
        roi_value = calculate_roi(nioDistrib, ioDistrib, num_bins)
        roi_dict[feature] = roi_value

    # filtered_features contains features with ROI values less than the threshold - high ROI features are not as good in distinguishing good/bad.
    # filtered_features = {feature: roi_value for feature, roi_value in roi_dict.items() if roi_value < threshold}
    # tsk_param.logger.debug('Features after overlap histogram using %d threshold:%s', threshold, filtered_features)

    # alternative to filtering features based on threshold, we can select the top n_dim_output features with the lowest ROI values
    top_n_elements = heapq.nsmallest(n_dim_output, roi_dict.items(), key=lambda item: item[1])
    filtered_features = dict(top_n_elements)

    extracted_columns_df = input_features_ft_df[filtered_features.keys()]

    return extracted_columns_df


def calculate_roi(nioDistrib, ioDistrib, num_bins):
    """
    This function checks how much overlap there is between the feature value distribution of good/bad signals (nioDistrib, ioDistrib)

    ROI value measures how much we can distinguish between good/bad signals as distributions.
        - ROI value is the region of interest - its the intersection/overlap area of the two histograms.
        - Small ROI means less overlap - better feature to use to distinguish between good/bad.
        - Big ROI means more overlap - not as good a feature to use to distinguish between good/bad.
    """
    # Calculate histograms for the non-interesting (nio) and interesting (io) distributions
    nioHist, nioBinEdges = np.histogram(nioDistrib, bins=num_bins)
    ioHist, ioBinEdges = np.histogram(ioDistrib, bins=num_bins)

    # Find the maximum of the minimum bin edges values for both distributions
    max_min = max(np.min(nioBinEdges), np.min(ioBinEdges))

    # Find the minimum of the maximum bin edges values for both distributions
    min_max = min(np.max(nioBinEdges), np.max(ioBinEdges))

    # Calculate the numerator (num) for the ROI
    num = np.max([0, min_max - max_min])

    # Find the minimum value among the bin edges values for both distributions
    minValue = min(np.min(nioBinEdges), np.min(ioBinEdges))

    # Find the maximum value among the bin edges values for both distributions
    maxValue = max(np.max(nioBinEdges), np.max(ioBinEdges))

    # Calculate the denominator (denom) for the ROI
    denom = np.abs(maxValue - minValue)

    # Calculate the ROI value
    roi = num / denom

    return roi


def find_outlier_indices(pca_df, method='dbscan', eps=0.5, min_samples=5, n_clusters=2):
    """
    Identifies the indices of outlier rows in the given DataFrame based on the specified clustering method.

    Parameters:
    - pca_df: 2D DataFrame where rows represent data points and columns represent features.
    - method: Clustering method to use ('dbscan' or 'kmeans'). Default is 'dbscan'.
    - eps: The maximum distance between two samples for them to be considered as in the same neighborhood (for DBSCAN).
    - min_samples: The number of samples in a neighborhood for a point to be considered a core point (for DBSCAN).
    - n_clusters: Number of clusters to form (for KMeans).

    Returns:
    - outlier_indices: A list of indices of the outlier rows.
    """
    if method == 'dbscan':
        eps, _ = calculate_eps(pca_df)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pca_df)
        # Identify outliers (DBSCAN labels outliers as -1)
        outlier_indices = np.where(clustering.labels_ == -1)[0]

    elif method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters).fit(pca_df)
        distances = np.min(clustering.transform(pca_df), axis=1)
        threshold = np.percentile(distances, 95)  # Consider top 5% as outliers
        outlier_indices = np.where(distances > threshold)[0]

    else:
        raise ValueError("Method must be 'dbscan' or 'kmeans'")

    return outlier_indices


def calculate_eps(data, k=4):
    """
    Calculates the optimal `eps` parameter using the elbow method.

    Parameters:
    - data: 2D numpy array or array-like structure where rows are data points and columns are features.
    - k: The number of nearest neighbors to consider (default is 4).

    Returns:
    - eps: Estimated `eps` value.
    - k_distances: The sorted k-distances used in the plot.
    """
    # Compute the nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(data)

    # Calculate the distances to the k-th nearest neighbor
    distances, indices = neighbors.kneighbors(data)

    # Sort the distances to the k-th nearest neighbor
    k_distances = np.sort(distances[:, k-1])

    # Use KneeLocator to find the "elbow" point
    knee_locator = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
    elbow_index = knee_locator.elbow

    # Calculate eps as the k-distance at the elbow point
    eps = k_distances[elbow_index]

    # Optionally, plot the elbow position
    # plot_elbow_position(k_distances, elbow_index)

    return eps, k_distances
