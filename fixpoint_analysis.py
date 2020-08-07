import numpy as np
import torch
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fix_points = None
data_file = '/home/falconinae/Documents/University/NDyn/RevRec/models/GRU_integration_07-08-2020_05-46-00_PM/fixpoints_07-08-2020_11-40-14_PM'
with open(data_file, 'rb') as file:
    fix_points = pickle.load(file)

fix_points_centered = fix_points - np.mean(fix_points, axis=0)


pca = PCA(n_components=3)
transformed_fixpoints = pca.fit_transform(fix_points_centered)
print('explained_variance: ', pca.explained_variance_ratio_)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(transformed_fixpoints[:,0], transformed_fixpoints[:,1], transformed_fixpoints[:,2], alpha=.2)
plt.show()
