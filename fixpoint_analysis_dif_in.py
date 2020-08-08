import numpy as np
import torch
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fix_points = None
data_file_0 = '/home/falconinae/Documents/University/NDyn/RevRec/models/GRU_integration_07-08-2020_05-46-00_PM/fixpoints_07-08-2020_11-40-14_PM'
with open(data_file_0, 'rb') as file:
    fix_points_0 = pickle.load(file)

data_file_1 = '/home/falconinae/Documents/University/NDyn/RevRec/models/GRU_integration_07-08-2020_05-46-00_PM/fixpoints_1_in_08-08-2020_10-48-41_AM'
with open(data_file_1, 'rb') as file:
    fix_points_1 = pickle.load(file)

data_file_2 = '/home/falconinae/Documents/University/NDyn/RevRec/models/GRU_integration_07-08-2020_05-46-00_PM/fixpoints_-1_in_08-08-2020_11-13-08_AM'
with open(data_file_2, 'rb') as file:
    fix_points_2 = pickle.load(file)

fix_points_all = np.concatenate((fix_points_0, fix_points_1, fix_points_2))
print(fix_points_all.shape)
fix_points_mean = np.mean(fix_points_all, axis=0)
fix_points_all_centered = fix_points_all - fix_points_mean
fix_points_0_centered = fix_points_0 - fix_points_mean
fix_points_1_centered = fix_points_1 - fix_points_mean
fix_points_2_centered = fix_points_2 - fix_points_mean

pca = PCA(n_components=3)
pca.fit(fix_points_all_centered)
print('explained_variance: ', pca.explained_variance_ratio_)
transformed_fixpoints_0 = pca.transform(fix_points_0_centered)[0:1000,:]
transformed_fixpoints_1 = pca.transform(fix_points_1_centered)[0:1000,:]
transformed_fixpoints_2 = pca.transform(fix_points_2_centered)[0:1000,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(transformed_fixpoints_0[:,0], transformed_fixpoints_0[:,1], transformed_fixpoints_0[:,2], alpha=.2, c='blue')
ax.scatter(transformed_fixpoints_1[:,0], transformed_fixpoints_1[:,1], transformed_fixpoints_1[:,2], alpha=.2, c='red')
ax.scatter(transformed_fixpoints_2[:,0], transformed_fixpoints_2[:,1], transformed_fixpoints_2[:,2], alpha=.2, c='green')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
