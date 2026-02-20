#----Eliminating warnings from scikit-learn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#--------------------------------------------------------------------------------------------------
def get_data(filepath):

#----Read the cleaned and normalized dataset from Labtask 2
    dataset = pd.read_csv(filepath)
#----Return the feature matrix (Volume, Doors) and the true style labels
    features = dataset[['Volume_Normalized', 'Doors_Normalized']].values
    styles = dataset['Style'].values
    return features, styles

#--------------------------------------------------------------------------------------------------
def do_cluster(number_of_clusters, features):

#----Define a KMeans model with the given number of clusters
    model = KMeans(n_clusters=number_of_clusters, random_state=0, n_init=10)
#----Fit the model to the feature data
    model.fit(features)
#----Get the cluster label assigned to each car
    labels = model.predict(features)
#----Each cluster has a center point in feature space
    centers = model.cluster_centers_
    return model, labels, centers

#--------------------------------------------------------------------------------------------------
def find_majority_style(cluster_styles):

#----Count how many cars of each style are in this cluster
    values, counts = np.unique(cluster_styles, return_counts=True)
#----The majority style is whichever style has the highest count
    majority_style = values[np.argmax(counts)]
    return majority_style

#--------------------------------------------------------------------------------------------------
def build_cluster_style_map(model, labels, styles):

#----Map each cluster number to its majority style
    cluster_style_map = {}
    for cluster_number in range(model.n_clusters):
#----Find all true styles belonging to cars in this cluster
        cluster_styles = styles[labels == cluster_number]
        cluster_style_map[cluster_number] = find_majority_style(cluster_styles)
    return cluster_style_map

#--------------------------------------------------------------------------------------------------
def save_cluster_cars(features, styles, labels, cluster_style_map, output_path):

#----Build a dataframe with Volume, Doors, true Style, and the ClusterStyle for each car
    results = pd.DataFrame({
        'Volume':   features[:, 0],
        'Doors':    features[:, 1],
        'Style':    styles,
        'ClusterStyle': [cluster_style_map[label] for label in labels]
    })
    results.to_csv(output_path, index=False)
    return results

#--------------------------------------------------------------------------------------------------
def compute_cluster_accuracy(model, labels, styles, cluster_style_map):

#----For each cluster, count the cars that match the cluster's majority style
    accuracy_rows = []
    for cluster_number in range(model.n_clusters):
#----All true styles for cars in this cluster
        cluster_styles = styles[labels == cluster_number]
        cluster_size = len(cluster_styles)
        majority_style = cluster_style_map[cluster_number]
#----Accuracy = number of cars with the majority style / total cars in cluster
        correct = np.sum(cluster_styles == majority_style)
        accuracy = correct / cluster_size if cluster_size > 0 else 0.0
        accuracy_rows.append({
            'ClusterStyle':  majority_style,
            'SizeOfCluster': cluster_size,
            'Accuracy':      round(accuracy, 4)
        })
    return pd.DataFrame(accuracy_rows)

#--------------------------------------------------------------------------------------------------
def save_cluster_accuracy(accuracy_df, output_path):

#----Write the accuracy table (one row per cluster) to CSV
    accuracy_df.to_csv(output_path, index=False)

#--------------------------------------------------------------------------------------------------
def plot_clusters(number_of_clusters, features, centers, labels, cluster_style_map):

#----Plot the clustered cars with a unique colour per cluster
    color = ['red', 'blue', 'green', 'purple', 'cyan']
    figure, plot_area = plt.subplots(figsize=(10, 6))
    for cluster_number in range(number_of_clusters):
        cluster_label = cluster_style_map[cluster_number]
        plot_area.scatter(
            features[labels == cluster_number, 0],
            features[labels == cluster_number, 1],
            c=color[cluster_number], s=20,
            label=f'Cluster {cluster_number}: {cluster_label}'
        )
#----Mark the center of each cluster with a square marker
        plot_area.plot(
            centers[cluster_number, 0], centers[cluster_number, 1],
            c=color[cluster_number], marker='s', markersize=10
        )
    plot_area.set_title('K-Means Clusters (5 Car Styles)', fontsize=20)
    plot_area.set_xlabel('Volume (Normalized)', fontsize=16)
    plot_area.set_ylabel('Doors (Normalized)', fontsize=16)
    plot_area.legend(fontsize=12)
    plt.draw()
    plt.show()
    plt.close()

#--------------------------------------------------------------------------------------------------
def main():

    number_of_clusters = 5
    data_path          = 'CleanedAndNormalizedFromLab2.csv'
    cars_output_path   = 'ClusterCars.csv'
    accuracy_output    = 'ClusterAccuracy.csv'

#----Load features and true style labels from the dataset
    features, styles = get_data(data_path)

#----Run K-Means with 5 clusters
    model, labels, centers = do_cluster(number_of_clusters, features)

#----Determine the majority (representative) style for each cluster
    cluster_style_map = build_cluster_style_map(model, labels, styles)

#----Save per-car results: Volume, Doors, Style, ClusterStyle
    save_cluster_cars(features, styles, labels, cluster_style_map, cars_output_path)
    print(f"Saved {cars_output_path}")

#----Compute and save per-cluster accuracy
    accuracy_df = compute_cluster_accuracy(model, labels, styles, cluster_style_map)
    save_cluster_accuracy(accuracy_df, accuracy_output)
    print(f"Saved {accuracy_output}")
    print(accuracy_df)

#----Visualise the clusters
    plot_clusters(number_of_clusters, features, centers, labels, cluster_style_map)

#--------------------------------------------------------------------------------------------------
main()
