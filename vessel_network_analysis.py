# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 15:11:36 2024

@author: Oleg Kovtun
"""
#%%

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('branch_info.csv')
df = df[df['Euclidean distance'] > 0]
skeleton_node_counts = df.groupby('Skeleton ID').size()
valid_skeleton_ids = skeleton_node_counts[skeleton_node_counts >= 15].index
df_filtered = df[df['Skeleton ID'].isin(valid_skeleton_ids)]

#%%

G = nx.Graph()

for index, row in df_filtered.iterrows():
    # V1 and V2 are the two nodes
    v1 = (row['V1 x'], row['V1 y'], row['V1 z'])
    v2 = (row['V2 x'], row['V2 y'], row['V2 z'])
    
    G.add_node(v1)
    G.add_node(v2)
    G.add_edge(v1, v2, length=row['Branch length'], distance=row['Euclidean distance'])


#%%

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt

tif_stack = tifffile.imread('MASK_microglia_ims.tif')
print('The TIF stack size is: ',tif_stack.shape)  

def compute_diameter_at_node(node_coords, tif_stack, radius=5):
    """
    Compute the vessel diameter at a given node's coordinates.
    
    Parameters:
    - node_coords: Tuple of (x, y, z) coordinates of the node
    - tif_stack: 3D binary numpy array representing the segmented vessels
    - radius: The radius of the cube around the node to sample
    
    Returns:
    - Estimated vessel diameter at the node
    """
    
    x, y, z = map(int, node_coords)
    x_min, x_max = max(0, x - radius), min(tif_stack.shape[2], x + radius)
    y_min, y_max = max(0, y - radius), min(tif_stack.shape[1], y + radius)
    z_min, z_max = max(0, z - radius), min(tif_stack.shape[0], z + radius)
    
    local_region = tif_stack[z_min:z_max, y_min:y_max, x_min:x_max]
    
    distance_map = distance_transform_edt(local_region == 255) 
    max_radius = np.max(distance_map)
    diameter = 2 * max_radius
    
    return diameter



for node in G.nodes():
    node_coords = node  
    diameter = compute_diameter_at_node(node_coords, tif_stack)
    G.nodes[node]['diameter'] = diameter


#%%

fig = plt.figure(figsize=(7,7),dpi=300)
ax = fig.add_subplot(111, projection='3d')    

pos = {node: node for node in G.nodes()}  
diameters = np.array([G.nodes[node]['diameter'] for node in G.nodes()])

norm = plt.Normalize(vmin=np.min(diameters), vmax=np.max(diameters))
cmap = plt.cm.get_cmap('viridis')  

for node, (x, y, z) in pos.items():
    diameter = G.nodes[node]['diameter']
    color = cmap(norm(diameter))  
    ax.scatter(x, y, z, color=color, s=50)  

for edge in G.edges():
    x_values = [pos[edge[0]][0], pos[edge[1]][0]]
    y_values = [pos[edge[0]][1], pos[edge[1]][1]]
    z_values = [pos[edge[0]][2], pos[edge[1]][2]]
    
    ax.plot(x_values, y_values, z_values, color='red') 

mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array(diameters)
fig.colorbar(mappable, ax=ax, label='Vessel Diameter')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()


#%%

from collections import defaultdict

bin_size = 100  

positions = np.array(list(G.nodes()))
min_coords = np.min(positions, axis=0)
max_coords = np.max(positions, axis=0)


x_bins = np.arange(min_coords[0], max_coords[0] + bin_size, bin_size)
y_bins = np.arange(min_coords[1], max_coords[1] + bin_size, bin_size)
z_bins = np.arange(min_coords[2], max_coords[2] + bin_size, bin_size)

vessel_density = defaultdict(list)
bifurcation_density = defaultdict(list)
diameter_mean = defaultdict(list)
diameter_std = defaultdict(list)

def get_bin_index(x, y, z):
    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1
    z_idx = np.digitize(z, z_bins) - 1
    return (x_idx, y_idx, z_idx)

node_bins = defaultdict(list)

for node in G.nodes():
    x, y, z = node  
    bin_index = get_bin_index(x, y, z)
    node_bins[bin_index].append(node)

for bin_index, nodes_in_bin in node_bins.items():
    if not nodes_in_bin:
        continue

    edges_in_bin = [(u, v) for u, v in G.edges(nodes_in_bin)]
    vessel_density[bin_index] = len(edges_in_bin)

    bifurcation_count = sum(1 for node in nodes_in_bin if G.degree[node] == 3)
    bifurcation_density[bin_index] = bifurcation_count

    diameters = [G.nodes[node]['diameter'] for node in nodes_in_bin if 'diameter' in G.nodes[node]]
    if diameters:
        diameter_mean[bin_index] = np.mean(diameters)
        diameter_std[bin_index] = np.std(diameters)

bin_index_example = (0, 0, 0)  # Example bin index
print(f"Mean diameter in bin {bin_index_example}: {diameter_mean[bin_index_example]} μm")

#%%

bins = list(vessel_density.keys())  

data = {
    'vessel_density': [vessel_density[bin] for bin in bins],
    'bifurcation_density': [bifurcation_density[bin] for bin in bins],
    'diameter_mean': [diameter_mean[bin] for bin in bins],
    'diameter_std': [diameter_std[bin] for bin in bins],
}

df_features = pd.DataFrame(data, index=bins)  

#%%

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

pca = PCA(n_components=2)  
pca_result = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=bins)

explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance (PC1, PC2): {explained_variance}")

plt.figure(figsize=(7,7),dpi=300)

bin_sums = [sum(bin_idx) for bin_idx in pca_df.index]

norm = plt.Normalize(vmin=np.min(bin_sums), vmax=np.max(bin_sums))
cmap = plt.cm.get_cmap('jet')  

sc = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=bin_sums, cmap=cmap, s=50)
plt.colorbar(sc, label='Segmentation Bin')

plt.title('PCA of Vascular Properties')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

for i, bin_idx in enumerate(pca_df.index):
    plt.annotate(bin_idx, (pca_df['PC1'][i], pca_df['PC2'][i]))

plt.show()


