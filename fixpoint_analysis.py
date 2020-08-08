import numpy as np
import torch
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FP_TOLERANCE = 0.000001

fix_points = None
data_file = 'models/GRU_integration_07-08-2020_12-36-02_PM/fixpoints_08-08-2020_12-08-04_PM'


# sort clipp Fps-04M'
with open(data_file, 'rb') as file:
    fix_points, fp_losses = pickle.load(file)

# sort clipp Fps
loss_idx = fp_losses < FP_TOLERANCE
#keep_idx = np.where(loss_idxss)[0]
fix_points_w_tol = fix_points[loss_idx]
fp_losses_w_tol = fp_losses[loss_idx]

# quality of fps before clipping

fig, axes = plt.subplots(2,1,figsize = (12,6), sharey=True, sharex=True)
axes[0].semilogy(np.sort(fp_losses))
axes[0].set_xlabel('Fp no clipping')
axes[0].set_ylabel('Fp loss')
axes[1].semilogy(np.sort(fp_losses_w_tol))
axes[1].set_xlabel('Fp with clipping')
axes[1].set_ylabel('Fp loss')
plt.suptitle("Quality of Fps")
#plt.show()

fix_points_centered = fix_points - np.mean(fix_points, axis=0)
print(fp_losses.shape)
print(fix_points.shape)

fix_points_w_tol_centered = fix_points_w_tol - np.mean(fix_points_w_tol, axis=0)





pca = PCA(n_components=3)
#transformed_fixpoints = pca.fit_transform(fix_points_centered)
transformed_fixpoints = pca.fit_transform(fix_points_w_tol_centered)
print(fp_losses.shape)
print(fix_points.shape)
color_loss = fp_losses_w_tol

print('explained_variance: ', pca.explained_variance_ratio_)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(transformed_fixpoints[:,0], transformed_fixpoints[:,1], transformed_fixpoints[:,2], alpha=.2, c=color_loss)
ax.set_ylabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.show()
