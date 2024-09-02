"""
Input: .tsv file of protein-protein interactions with a 'Source' and 'Target' column
Output: Principal component analysis scatterplot of 12 centrality measures
- PCA scatterplot with clusters
- PCA loadings plot
- Kmeans elbow curve (to determine the amount of clusters)
- csv file containing all data
"""

import os
import pandas as pd
import networkx as nx
import netcenlib as ncl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from palettable.colorbrewer.qualitative import Set3_12

# Specify file paths

ppi_path: str = 'ava_predictions_data/TS5Scores_KSHV_AVA_results_all_ipTM_0-47_forgephi.tsv'

# Read .csv files and initialize dataframes
ppi_df: pd.DataFrame = pd.read_csv(ppi_path, skipinitialspace=True, sep='\t')

# Create graph
G = nx.from_pandas_edgelist(ppi_df, 'Source', 'Target')

print(G)

# Put all the nodes into the output dataframe
df = pd.DataFrame(G.nodes, columns=['Name'])

# Calculate centrality measures and append to output dataframe
# For a start 21 centrality measures were taken from a paper by Khojasteh et al., Nature Scientific Reports (2022)
# https://www.nature.com/articles/s41598-022-08574-6

# --- 21 Paper centrality measures ---
# Average Distance, Barycenter, Closeness (Freeman), Closeness (Latora), Residual closeness,
# Decay, Diffusion degree, Geodesic K-Path, Laplacian,
# Leverage, Lin, Lobby, Markov, Radiality,
# Eigenvector, Subgraph scores, Shortest-Paths betweenness, Eccentricity, Degree,
# Kleinberg’s authority scores, and Kleinberg’s hub scores.
# ----------------------------------

# Unfortunately, not all centrality measures the authors used are in the netcenlib package and I could not get some of them to work. Thus, just the available ones were chosen.
# netcenlib was released just in January 2024. Perhaps one could contribute by implementing more algorithms from centiserver!
# Betweenness was added as a centrality measure.
# --- 12 Chosen centrality measures ---
# Average Distance, Closeness (Freeman),
# Decay, Diffusion degree, Geodesic K-Path, Laplacian,
# Leverage, Lin,
# Eigenvector, Subgraph scores, Degree and Betweenness
# ----------------------------------

# Average Distance
average_distance_centrality = ncl.average_distance_centrality(G)
df['Average Distance Centrality'] = df['Name'].map(average_distance_centrality)

# Closeness (Freeman)
closeness_centrality = ncl.closeness_centrality(G)
df['Closeness Centrality'] = df['Name'].map(closeness_centrality)

# Decay
decay_centrality = ncl.decay_centrality(G)
df['Decay Centrality'] = df['Name'].map(decay_centrality)

# Diffusion Degree
diffusion_degree_centrality = ncl.diffusion_degree_centrality(G)
df['Diffusion Degree Centrality'] = df['Name'].map(diffusion_degree_centrality)

# Geodestic K Path
geodestic_k_path_centrality = ncl.geodestic_k_path_centrality(G)
df['Geodestic K Path Centrality'] = df['Name'].map(geodestic_k_path_centrality)

# Laplacian
laplacian_centrality = ncl.laplacian_centrality(G)
df['Laplacian Centrality'] = df['Name'].map(laplacian_centrality)

# Leverage
leverage_centrality = ncl.leverage_centrality(G)
df['Leverage Centrality'] = df['Name'].map(leverage_centrality)

# Lin
lin_centrality = ncl.lin_centrality(G)
df['Lin Centrality'] = df['Name'].map(lin_centrality)

# Eigenvector
eigenvector_centrality = ncl.eigenvector_centrality(G)
df['Eigenvector Centrality'] = df['Name'].map(eigenvector_centrality)

# Subgraph
subgraph_centrality = ncl.subgraph_centrality(G)
df['Subgraph Centrality'] = df['Name'].map(subgraph_centrality)

# Betweenness
betweenness_centrality = ncl.betweenness_centrality(G)
df['Betweenness Centrality'] = df['Name'].map(betweenness_centrality)

# Degree
degree_centrality = ncl.degree_centrality(G)
df['Degree Centrality'] = df['Name'].map(degree_centrality)


# --- PCA ---

# Extract features
feature_names = df.drop(columns=['Name']).columns.to_list()
features = df.drop(columns=['Name']).values

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Create PCA object with two principal components
pca = PCA(n_components=2)

# Perform PCA
principal_components = pca.fit_transform(scaled_features)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=0)  # Choose the number of clusters, ideally via the elbow method
clusters = kmeans.fit_predict(principal_components)

# Add cluster labels to dataframe
df['Cluster'] = clusters

principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

df['PC1'] = principal_df['PC1']
df['PC2'] = principal_df['PC2']

df.to_csv('topology.csv', index=False)

# --- Elbow Method K means cluster numbers ---
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(principal_components)
    wcss.append(kmeans.inertia_)  # inertia_ gives the WCSS

# Plotting the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.xticks(range(1, 11))
plt.grid(True)
plt.savefig('kmeans_elbow_curve.png')

# --- PCA Scatterplot with clusters ---

# Extract explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
variance_captured = explained_variance_ratio[0] + explained_variance_ratio[1]

# Create figure with two columns and one row
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

sns.scatterplot(ax=axes[0], x='PC1', y='PC2', hue='Cluster', data=df, palette='Set1', s=100)

# Plot PCA scatterplot on the first subplot
axes[0].set_title('(A) PCA Scatterplot, ' + 'variance captured: ' + str(round(variance_captured, 2)))
axes[0].set_ylabel('Principal Component 1')
axes[0].set_xlabel('Principal Component 2')
axes[0].grid(True)
axes[0].legend(title='Cluster', loc='upper right')

"""
# Annotate cluster centers or representative points
cluster_centers = kmeans.cluster_centers_
for i, center in enumerate(cluster_centers):
    axes[0].annotate(f'Cluster {i}', (center[0], center[1]), textcoords="offset points", xytext=(0,10), ha='center')
"""

# Annotate points with their labels
for index, row in df.iterrows():
    point = (row['PC2'], row['PC1'])
    label = row['Name']
    
    # Generate random angle for the arrow
    angle = np.random.uniform(0, 2*np.pi)
    
    # Fix the length of the arrow
    arrow_length = 40
    
    # Calculate the x and y components of the offset vector using the random angle
    offset_x = np.cos(angle) * arrow_length
    offset_y = np.sin(angle) * arrow_length
    
    # Conditions to label a subset of points based on pc1 and pc2 coordinates, currently set to impossible value to label no points
    if row['PC1'] == -1 and row['PC2'] == -1:
        axes[0].annotate(text=label, xy=point, xytext=(offset_y, offset_x), textcoords='offset points',
                    arrowprops=dict(arrowstyle='-', color='black', linewidth=0.5))


# --- PCA loadings as quiver arrows on the second subplot ---
# coordinates of features (i.e., loadings; note the transpose)
loadings = pca.components_[:2].T.tolist()

for i in range(len(loadings)):
    loadings[i].append(features[i])

# Get a list of 13 compatible ColorBrewer colors
colors = Set3_12.hex_colors


for index, loading in enumerate(loadings):
    color = colors[index]

    q = plt.quiver(
        0,
        0,
        loading[0],
        loading[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        label=feature_names[index]
    )

axes[1].set_xlim(-1, 1)
axes[1].set_ylim(-1, 1)
axes[1].set_xlabel("First principal component loading")
axes[1].set_ylabel("Second principal component loading")
axes[1].set_title("(B) PCA Loadings Plot")
axes[1].legend(title='Centrality-score')
axes[1].set_aspect("equal")

plt.tight_layout()
plt.savefig('pca.png')