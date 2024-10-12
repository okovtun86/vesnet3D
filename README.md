# vesnet3D
## 3D Vascular Network Analysis

The vessel_network_analysis.py script imports FIJI-generated 3D skeleton information to create a graph representation in 3D, calculates vessel diameter for each node, computes 4 features for nodes within pre-defined network bins (vessel density, bifurcation density, diameter mean, diameter std), and runs 2-component PCA on the resulting feature dataframe. 

1. Open the 3D .tif stack in FIJI, pre-process (e.g., adaptive histogram equalization + median filtering), and threshold to generate the binary 3D stack. Save it.
2. Skeletonize the binary stack in 3D via Plugins › Skeleton › Skeletonize (2D/3D).
3. Generate skeleton information via Analyze › Skeleton › Analyze Skeleton (2D/3D) (none for loop elimination, only 'detailed info' checked). Save the resulting Branch Information table as a .csv file.
4. Analyze the branch_info.csv and binary mask .tif stack using the vessel_network_analysis.py script.

