import json
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns

"""
This file contains functions that extract data from different object primarily aimed at creating logging and reporting data.

"""


def dataframe_metadata(df, add_data_types=True, add_memory_usage=True, df_name='This'):
    """
    Extracts metadata from a dataframe.

    :param df: input dataframe to extract metadata from.

    :return: summary of metadata
    :return type: str

    """
    shape_info = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
    columns_info = f"Column Names: {', '.join(df.columns)}"

    dtypes_info = ''
    if add_data_types:
        dtypes_info = "\n\t\t\t\t\t\tData Types:" + "".join([f"\n\t\t\t\t\t\t\t{col}: {dtype}" for col, dtype in df.dtypes.items()])

    memory_info = ''
    if add_memory_usage:
        memory_info = f"\n\t\t\t\t\t\tMemory Usage: {df.memory_usage(deep=True).sum()} bytes"

    # Combine all information into a single string
    metadata = f"{df_name} dataFrame metadata:\n\t\t\t\t\t\t{shape_info}\n\t\t\t\t\t\t{columns_info}{dtypes_info}{memory_info}"

    return metadata


def model_metadata(model):
    """
    Extracts metadata from a model object.

    :param model: Input model to extract metadata from.

    :return: Model data summary
    :return type: str

    """

    model_name = f"model_name: {model.model_name}"
    model_metadata = json.dumps(model.model_metadata, indent=4)

    # Combine all information into a single string
    metadata = f"Model Metadata:\n\t\t\t\t\t\t{model_name}\n{model_metadata}"

    return metadata


def histogram_overlap_plot(ft_df):
    """
    Takes a datframe and plots a histogram of all columns (features) on a single plot

    """

    # Assuming `df` is the DataFrame with the features as columns

    matplotlib.use('Agg')

    # Number of features
    num_features = ft_df.shape[1]

    # Create a color map using a perceptually uniform color map from matplotlib
    cmap = plt.cm.get_cmap('viridis', num_features)
    colors = [cmap(i) for i in range(num_features)]

    # Plot all features in a single histogram
    plt.figure(figsize=(12, 8))

    # Iterate through each feature
    for i, col in enumerate(ft_df.columns[:10]):
        print(col)
        plt.hist(ft_df[col], bins=100, alpha=0.5, color=colors[i], label=col)

    # Adding legend (limited to avoid clutter)
    max_legend_items = 10
    plt.legend(title='Features', loc='upper right', ncol=2, fontsize=8, labels=ft_df.columns[:max_legend_items])
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.title('Histograms of Feature Values by Series')
    plt.grid(axis='y', alpha=0.75)

    # Show the plot
    plt.show()


def plot_pca_2d(df, targetFilename, targetFolder, title='', label_column='label', columns_to_ignore=['cycle_id']):
    """
    Plots a 2D PCA of the dataframe while ignoring the specified columns and colors
    the data points based on the label column values, if present.

    :param df: The input dataframe containing the data.
    :type df: pandas.DataFrame
    :param label_column: The name of the column containing labels for colouring the points.
    :type label_column: str
    :param columns_to_ignore: List of column names to ignore for PCA, except the label column.
    :type columns_to_ignore: list of str
    """

    matplotlib.use('Agg')

    # Drop the columns that should not be used in PCA, except the label column if it exists
    columns_to_ignore = [col for col in columns_to_ignore if col in df.columns and col != label_column]
    df_reduced = df.drop(columns=columns_to_ignore, errors='ignore')

    # Check if the label column exists in the dataframe
    if label_column in df_reduced.columns:
        labels = df_reduced[label_column].values
        df_reduced = df_reduced.drop(columns=[label_column])

        colors = ['blue' if label == 1 else 'red' for label in labels]

    else:
        colors = 'grey'

    # Initialize PCA and fit-transform the data
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_reduced)

    # Create a scatter plot of the PCA result with colored points
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, edgecolors='k', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(targetFolder, targetFilename + '.png'))
    plt.close()
    # plt.show()


def plot_upper_triangular_heatmap(corr_matrix, target_filename, target_folder, title=''):
    """
    Plots a heatmap for the strictly upper triangular part of a correlation matrix and saves the figure.

    Parameters:
    - corr_matrix: Pandas DataFrame containing the correlation matrix.
    - title: Title of the heatmap.
    - target_filename: Name of the file to save the plot (e.g., 'heatmap.png').
    - target_folder: Directory where the file should be saved.
    """

    matplotlib.use('Agg')

    # Mask the lower triangle and the diagonal
    # mask = np.triu(np.ones_like(corr_matrix, dtype=bool)).T

    # Create the heatmap
    plt.figure(figsize=(20, 15))
    ax = sns.heatmap(corr_matrix, cmap='jet', cbar=True, square=True, vmin=0, vmax=1)

    # Add grid lines to the heatmap
    ax.grid(True, linestyle='-', linewidth=0.7, color='black')

    # Rotate labels for better readability
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    # Adding title and labels
    plt.title(title, fontsize=16)
    plt.xlabel('Columns', fontsize=12)
    plt.ylabel('Rows', fontsize=12)

    # Save the figure
    file_path = os.path.join(target_folder, target_filename)
    plt.savefig(file_path, bbox_inches='tight')

    plt.close()

    # Show the heatmap
    # plt.show()


def plot_correlation_histogram(correlation_df, target_filename='', target_folder='', top_value=0.95):
    """
    Plots three histograms of correlation values:
    - Highest negative correlations
    - Highest positive correlations
    - Full range of correlations (below the other two)

    Parameters:
        correlation_df (pd.DataFrame): DataFrame with a single row of correlation values.
        top_n (int): Number of top correlated features to highlight.
    """

    matplotlib.use('Agg')

    # Ensure the DataFrame has a single row
    if len(correlation_df) != 1:
        raise ValueError("The DataFrame should have exactly one row.")

    # Get the correlation values
    correlations = correlation_df.iloc[0]
    top_neg_corr = correlations[correlations < -top_value]
    top_pos_corr = correlations[correlations >= top_value]

    # Set up the figure for three plots
    fig = plt.figure(figsize=(14, 12))

    # Plot for negative correlations in the first subplot (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(top_neg_corr, bins=100, edgecolor='k', alpha=0.7)
    ax1.set_title('Highest Negative Correlations')
    ax1.set_xlabel('Correlation Value')
    ax1.set_ylabel('Frequency')

    # Plot for positive correlations in the second subplot (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(top_pos_corr, bins=100, edgecolor='k', alpha=0.7)
    ax2.set_title('Highest Positive Correlations')
    ax2.set_xlabel('Correlation Value')

    # Plot for full range of correlations spanning the entire bottom row
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.hist(correlations, bins=100, edgecolor='k', alpha=0.7)
    ax3.set_title('Full Range of Correlations')
    ax3.set_xlabel('Correlation Value')
    ax3.set_ylabel('Frequency')

    plt.tight_layout()

    file_path = os.path.join(target_folder, target_filename)
    plt.savefig(file_path, bbox_inches='tight')

    # plt.show()


def format_feature_list(features):
    """
    Formats the features and their values as a string.
    """
    return "\n".join(f"{column}: {value:.2f}" for column, value in features.items())


def correlated_features_summary(df, top_n=5):
    """
    Returns a string with the column names and values for the top and bottom N features,
    as well as the N features closest to zero in a single-row DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with a single row of feature values.
        top_n (int): Number of top, bottom, and closest-to-zero features to display.

    Returns:
        str: Formatted string with top, bottom, and closest-to-zero features.
    """
    # Ensure the DataFrame has a single row
    if len(df) != 1:
        raise ValueError("The DataFrame should have exactly one row.")

    # Get the values and columns
    values = df.iloc[0]

    # Get the top and bottom N values
    top_features = values.nlargest(top_n)
    bottom_features = values.nsmallest(top_n)

    # Get the top N features closest to zero
    closest_to_zero = values.abs().nsmallest(top_n)
    closest_features = values[closest_to_zero.index]

    # Format the results
    result = "Most Correlated Features:\n" + format_feature_list(top_features)
    result += "\n\nMost Anticorrelated Features:\n" + format_feature_list(bottom_features)
    result += "\n\nLeast Correlated Features:\n" + format_feature_list(closest_features)

    return result
