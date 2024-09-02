"""
Input: .tsv edgelist of proteins with 'source' and 'target' columns
Uses matplotlib to plot the node degree on y-axis and node names on the x-axis
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx
import os


# Specify filepath
file = 'ava_predictions_data/TS5Scores_HSV-1_AVA_results_all_ipTM_0-47_forgephi.tsv'

edgelist = pd.read_csv(file, sep='\t', skipinitialspace=True)
graph = nx.from_pandas_edgelist(edgelist, 'Source', 'Target')
degree_data = nx.degree(graph)
sorted_degree_data = sorted(degree_data, key=lambda x: x[1])
proteins, degrees = zip(*sorted_degree_data)

output_df = pd.DataFrame({'Protein': proteins, 'Degree': degrees})
output_df.to_csv('degree.csv', index=False)

fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(proteins, degrees)
# Show protein names with rotated labels under each bar
ax.set_xticklabels(proteins, rotation=90, ha='right')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylabel('Degree')
plt.tight_layout()
plt.savefig('degree_barplot.png')