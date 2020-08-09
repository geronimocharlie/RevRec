import numpy as np
import torch
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rnns import RNN, GRU, LSTM
from scipy.spatial.distance import pdist, squareform




def clip_fixpoints(fix_points, fp_losses, tolerance):
    #clip Fps
    loss_idx = fp_losses < FP_TOLERANCE
    #keep_idx = np.where(loss_idxss)[0]
    fix_points_w_tol = fix_points[loss_idx]
    fp_losses_w_tol = fp_losses[loss_idx]
    return fix_points_w_tol, fp_losses_w_tol

def oulier_removal(fix_points, fp_losses, outlier_dis, metric='eucledian'):

    #distances =
    pass

def keep_unique_fix_points(fix_points, fp_losses):
    pass

def plot_fp_qualitiy(fp_losses_original, fp_losses_clippd, show=False):
    # quality of fps
    fig, axes = plt.subplots(2,1,figsize = (12,6), sharey=True, sharex=True)
    axes[0].semilogy(np.sort(fp_losses_original))
    axes[0].set_xlabel('Fp no clipping')
    axes[0].set_ylabel('Fp loss')
    axes[1].semilogy(np.sort(fp_losses_clippd))
    axes[1].set_xlabel('Fp with clipping')
    axes[1].set_ylabel('Fp loss')
    plt.suptitle("Quality of Fps")
    if show:
        plt.show()
    else: return fig


def get_read_out_projection(fix_points, model, show=False):
    fix_points = torch.from_numpy(np.expand_dims(fix_points, axis=1))
    print(fix_points.size(), 'fixoints shape')
    activations = torch.sigmoid(model.read_out(fix_points))
    activations = activations.detach().numpy()
    print('Read out projection:')
    print(np.max(activations))
    print(np.min(activations))
    print(activations.shape)
    fig = plt.subplot()
    fig.plot(np.sort(np.squeeze(activations)))
    plt.xlabel('Fix points')
    plt.ylabel('Read out activation')
    plt.suptitle('Projection of fix points on read out layer')
    #fig.set_ylim(0,1)
    if show:
        plt.show()
    return activations, fig


def plot_fixpoints(fix_points, color, show=False):

    fix_points_centered = fix_points - np.mean(fix_points, axis=0)
    pca = PCA(n_components=3)
    transformed_fixpoints = pca.fit_transform(fix_points_centered)
    print('explained_variance: ', pca.explained_variance_ratio_)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(transformed_fixpoints[:,0], transformed_fixpoints[:,1], transformed_fixpoints[:,2], alpha=.2, c=color)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    fig.colorbar(im, ax=ax)
    if show:
        plt.show()
    else: return fig


if __name__=='__main__':

    FP_TOLERANCE = 1e-13
    model_directory = 'models/GRU_integration_09-08-2020_03-13-31_PM/'
    fp_file = 'fixpoints_09-08-2020_03-15-14_PM'
    model_file = 'trained_weights_GRU_integration_epochs_5'

    with open(f'{model_directory}{fp_file}', 'rb') as file:
        fix_points, fp_losses = pickle.load(file)

    model = torch.load(f'{model_directory}{model_file}')


    fix_points_c, fp_losses_c = clip_fixpoints(fix_points, fp_losses, FP_TOLERANCE)
    fig1 = plot_fp_qualitiy(fp_losses, fp_losses_c)
    fp_readout, _ = get_read_out_projection(fix_points_c, model)
    fig3 = plot_fixpoints(fix_points_c, fp_readout)
    plt.show()
