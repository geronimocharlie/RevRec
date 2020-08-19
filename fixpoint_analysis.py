from hyperparameters import *
import numpy as np
import torch
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import  random
from rnns import RNN, GRU, LSTM
from scipy.spatial.distance import pdist, squareform

def plot_with_runs( fix_points, runs, fp_color='g', runs_color='r', num_runs='all'):
    """
    Plotting fix points and trajectories in one 3d plot.

    @params:
        fix_points [np.array] (num_fixpoints, hidden_size): array of fix points to plot
        runs [np.array] (num_runs, sequence_lenght, hidden_size): array of trajectories over time
        fp_color [str] or [np.array](num_fixpoints, 1, output_size): default color or value of read out activation
        runs_color [str] or [np.array] (num_runs, seq_length, output_sze): default color or value of read activation of everey point in the run trajectory
        num_runs [str] or [int]: plotting all trajectories per default, may specifiy number of trajectories to plot
    """

    fix_points_centered = fix_points - np.mean(fix_points, axis=0)

    pca = PCA(n_components=3)
    transformed_fixpoints = pca.fit_transform(fix_points_centered)
    print('explained_variance: ', pca.explained_variance_ratio_)

    if num_runs == 'all':
        runs_list = runs
        runs_c_list = runs_color
    else:
        # choose random choice of tarjectories to plot
        random.seed(a=12)
        runs_list = random.sample(np.split(runs, runs.shape[0]), k=num_runs)

        if not(isinstance(runs_color, str)):
            random.seed(a=12)
            runs_c_list = random.sample(np.split(runs_color, runs_color.shape[0]), k=num_runs)
        else:
            runs_c_list = runs_color

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot fix points with read out color
    ax.scatter(transformed_fixpoints[:,0], transformed_fixpoints[:,1], transformed_fixpoints[:,2], alpha=.5, c=fp_color)

    # plot trajectories and read out color
    for r,c in zip(runs_list, runs_c_list):
        run = pca.transform(np.squeeze(r))
        ax.scatter(xs=run[:,0], ys=run[:,1], zs=run[:,2], c=c, alpha=0.2, marker='|')

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')

    return fig


def clip_fixpoints(fix_points, fp_losses, tolerance=0.0001):
    """
    Clipping fix points which losses are greater than the specified tolerance.
    """
    #clip Fps
    loss_idx = fp_losses < tolerance
    #keep_idx = np.where(loss_idxss)[0]
    fix_points_w_tol = fix_points[loss_idx]
    fp_losses_w_tol = fp_losses[loss_idx]
    return fix_points_w_tol, fp_losses_w_tol


def oulier_removal(fix_points, fp_losses, outlier_dis=1, print=False):
    #TODO: not yet working
    """
    Remove Outliers (points whose closes neighbur is further away than the threshold)
    """
    # pairwise difference between all fix points
    distances = squareform(pdist(fix_points))

    # find closest neighbur for each fix point -> smallest element in each column
    closest_neighbor = np.partition(distances, axis=0)[1]

    keep_idx = np.where(closest_neighbor < outlier_dist[0])
    fix_points_keep = fix_points[keep_idx]
    fp_losses_keep = fp_losses[keep_idx]

    if print:
        print(f"Removed {len(fix_points) - len(fix_points_keep)} points")

    return fix_points_keep, fp_losses_keep


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
    # TODO: adjust to flip flop / 3d task in general
    """
    @params:
        fix_points [np.array] (num_fixpoints, hidden_size)
        model [nn.Module]
        show [bool]: showing activation plot
    """

    fix_points = torch.from_numpy(np.expand_dims(fix_points, axis=1))
    activations = torch.sigmoid(model.read_out(fix_points))
    activations = activations.detach().numpy()


    fig = plt.subplot()
    fig.plot(np.sort(np.squeeze(activations)))
    plt.xlabel('Fix points')
    plt.ylabel('Read out activation')
    plt.suptitle('Projection of fix points on read out layer')
    if show:
        plt.show()

    return activations


def get_read_out_projection_run(runs, model):
    # TODO: adjust to flip flop task as well
    """
    Getting a read out activation for each point in a trajectory.

    @params:
        runs [np.array](num_runs, lenght, hidden_size): array of runs
        model [nn.Module]: model object to provide activations

    @returns:
        activations [np.array] (num_runs, seq_length, out_size)
    """
    runs = torch.from_numpy(runs)
    activations = torch.sigmoid(model.read_out(runs))
    activations = activations.detach().numpy()
    return activations


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


def load_fixpoints_and_losses():
    """
    @returns:
        fix_points [np.array] (num_fixpoints, hidden_size): array of fix points
        fix_points_losses [np.array] (num_fixpoints, 1): array of losses for each fix point
    """
    fps = []
    fp_ls = []
    for f in FIX_POINT_FILE:
        with open(f'{MODEL_PATH}{f}', 'rb') as file:
            fix_points, fp_losses = pickle.load(file)
            fps.append(fix_points)
            fp_ls.append(fp_losses)

    fix_points = np.concatenate(fps)
    fix_points_losses = np.concatenate(fp_ls)
    return fix_points, fix_points_losses
    

def load_fixpoints_and_losses_with_readout():
    """
    @returns:
        fix_points [np.array] (num_fixpoints, hidden_size): array of fix points
        fix_points_losses [np.array] (num_fixpoints, 1): array of losses for each fix point
        fix_points_readout [np.array] (num_fixpoints, 1, output_size): array of read out activation for each fix points
    """
    fps = []
    fp_ls = []
    for f in FIX_POINT_FILE:
        with open(f'{MODEL_PATH}{f}', 'rb') as file:
            fix_points, fp_losses = pickle.load(file)
            fps.append(fix_points)
            fp_ls.append(fp_losses)

    fix_points = np.concatenate(fps)
    fix_points_losses = np.concatenate(fp_ls)

    fp_r = []
    for f in fps:
        fp_readout = get_read_out_projection(f, model)
        fp_r.append(fp_readout)

    fix_points_readout = np.concatenate(fp_r)

    return fix_points, fix_points_losses, fix_points_readout

def load_runs():
    """
    Loadnig all the saved trajectories including read out projection.

    @returns:
        runs [np.array] (num_runs, sequence_length, hidden_size): array of all runs
        runs_readout [np.array] (num_runs, sequence_length, output_size): read out activation for each ponit of the run
    """
    runs = []
    for f in RUN_FILE:
        with open(f'{MODEL_PATH}{f}', 'rb') as file:
            run = pickle.load(file)
            runs.append(run)
    r_r = []
    for r in runs:
        r_readout = get_read_out_projection_run(r, model)
        r_r.append(r_readout)

    if len(runs)>0:
        runs = np.concatenate(runs)
        runs_readout = np.concatenate(r_r)
    else:
        runs = None
        runs_readout = None

    return runs, runs_readout



if __name__=='__main__':
    mode = 'charlie'
    #mode = 'leon'

    if mode == 'charlie':
        FP_TOLERANCE = 1e-14

        # load model
        model = torch.load(f'{MODEL_PATH}{MODEL_NAME}')

        # load fixpoints and read_outs
        fps, fps_ls, fps_rs = load_fixpoints_and_losses_with_readout()

        # if provided load trajectories
        if len(RUN_FILE)>0:
            runs, runs_r = load_runs()

        # clipping of fix points and analyzing fix point quality
        fps_c, fps_ls_c = clip_fixpoints(fps, fps_ls, FP_TOL)
        print(f"removed {fps.shape[0] - fps_c.shape[0]} fix points during clipping")
        #fix_points_c, fp_losses = oulier_removal(fix_points_c, fp_losses_c )
        fig1 = plot_fp_qualitiy(fps_ls, fps_ls_c)

        # plotting fix points (optioanlly with runs)
        if len(RUN_FILE)>0:
            fig = plot_with_runs(fps, runs, fps_rs, runs_r) #fps_rs, runs_r)
        else:
            fig = plot_fixpoints(fps, fps_rs)

        plt.show()



    if mode == 'leon':
        plot_with_runs('/home/falconinae/Documents/University/NDyn/RevRec/models/GRU_integration_07-08-2020_05-46-00_PM/fixpoints_07-08-2020_11-40-14_PM', '/home/falconinae/Documents/University/NDyn/RevRec/models/GRU_integration_07-08-2020_05-46-00_PM/exemplary_runs_09-08-2020_02-32-55_PM')
