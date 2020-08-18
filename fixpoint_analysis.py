
import numpy as np
import torch
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import  random
def plot_with_runs(self, fix_points_file, runs_file):
    NUM_RUNS = 3
    fix_points = None
    data_file =
    with open(data_file, 'rb') as file:
        fix_points = pickle.load(file)

    fix_points_centered = fix_points - np.mean(fix_points, axis=0)


    pca = PCA(n_components=3)
    transformed_fixpoints = pca.fit_transform(fix_points_centered)
    print('explained_variance: ', pca.explained_variance_ratio_)

    #get some exemplary_runs to also plot
    runs_file = runs_file
    with open(runs_file, 'rb') as file:
        runs = pickle.load(file)
    runs_list = random.sample(np.split(runs, runs.shape[0]), k=NUM_RUNS)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed_fixpoints[:,0], transformed_fixpoints[:,1], transformed_fixpoints[:,2], alpha=.2)
    for run in runs_list:
        run = pca.transform(np.squeeze(run))
        print(run.shape)
        ax.plot(xs=run[:,0], ys=run[:,1], zs=run[:,2], c='cyan')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == '__main__':
    plot_with_runs('/home/falconinae/Documents/University/NDyn/RevRec/models/GRU_integration_07-08-2020_05-46-00_PM/fixpoints_07-08-2020_11-40-14_PM', '/home/falconinae/Documents/University/NDyn/RevRec/models/GRU_integration_07-08-2020_05-46-00_PM/exemplary_runs_09-08-2020_02-32-55_PM')
