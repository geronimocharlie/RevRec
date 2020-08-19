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

def plot_with_runs( fix_points, fp_color, runs, runs_color):

    # plot all runs
    # plot with readout on fix_points and on runs
    NUM_RUNS = 2
    #print(runs.shape, "shape of runs")
    fix_points_centered = fix_points - np.mean(fix_points, axis=0)


    pca = PCA(n_components=3)
    transformed_fixpoints = pca.fit_transform(fix_points_centered)
    print('explained_variance: ', pca.explained_variance_ratio_)

    #get some exemplary_runs to also plot
    random.seed(a=12)
    runs_list = random.sample(np.split(runs, runs.shape[0]), k=NUM_RUNS)
    random.seed(a=12)
    runs_c_list = random.sample(np.split(runs_color, runs_color.shape[0]), k=NUM_RUNS)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed_fixpoints[:,0], transformed_fixpoints[:,1], transformed_fixpoints[:,2], alpha=.5, c=fp_color)

    #for run in runs_list:
    #    run = pca.transform(np.squeeze(run))
    #    print(run.shape)
    for r,c in zip(runs_list, runs_c_list):
        print(r.shape, "r.shape")
        run = pca.transform(np.squeeze(r))
        ax.scatter(xs=run[:,0], ys=run[:,1], zs=run[:,2], c=c, alpha=0.2, marker='|')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    return fig




def clip_fixpoints(fix_points, fp_losses, tolerance=0.0001):
    #clip Fps
    loss_idx = fp_losses < tolerance
    #keep_idx = np.where(loss_idxss)[0]
    fix_points_w_tol = fix_points[loss_idx]
    fp_losses_w_tol = fp_losses[loss_idx]
    return fix_points_w_tol, fp_losses_w_tol

def oulier_removal(fix_points, fp_losses, outlier_dis=1, print=False):
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


def get_read_out_projection(fix_points, model, show=True):
    fix_points = torch.from_numpy(np.expand_dims(fix_points, axis=1))
    #print(fix_points.size(), 'fixoints shape')
    activations = torch.sigmoid(model.read_out(fix_points))
    activations = activations.detach().numpy()
    #print('Read out projection:')
    #print(np.max(activations))
    #print(np.min(activations))
    #print(activations.shape)
    fig = plt.subplot()
    fig.plot(np.sort(np.squeeze(activations)))
    plt.xlabel('Fix points')
    plt.ylabel('Read out activation')
    plt.suptitle('Projection of fix points on read out layer')
    #fig.set_ylim(0,1)
    if show:
        plt.show()
    return activations, fig

def get_read_out_projection_run(runs, model, show=False):
    "runs (num_runs, lenght, hidden_size)"
    runs = torch.from_numpy(runs)
    activations = torch.sigmoid(model.read_out(runs))
    activations = activations.detach().numpy()
    #activations = np.expand_dims(activations, 0)
    #print("run read out activations shape", activations.shape)
    #fig = plt.subplot()
    #fig.plot(np.sort(np.squeeze(activations)))
    #plt.xlabel('Fix points')
    #plt.ylabel('Read out activation')
    #plt.suptitle('Projection of fix points on read out layer')
    #if show:
    #    plt.show()
    return activations#, fig



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
    mode = 'charlie'
    #mode = 'charlie'
    if mode == 'charlie':
        #FP_TOLERANCE = 1e-14

        fps = []
        fp_ls = []
        for f in FIX_POINT_FILE:
            with open(f'{MODEL_PATH}{f}', 'rb') as file:
                fix_points, fp_losses = pickle.load(file)
                fps.append(fix_points)
                fp_ls.append(fp_losses)

        model = torch.load(f'{MODEL_PATH}{MODEL_NAME}')

        runs = []
        for f in RUN_FILE:
            with open(f'{MODEL_PATH}{f}', 'rb') as file:
                run = pickle.load(file)
                runs.append(run)
        #print(fix_points.shape, "fp shape")

        #fix_points_c, fp_losses_c = clip_fixpoints(fix_points, fp_losses, FP_TOL)
        #print(fix_points_c.shape, "fp clipped shape")
        #fix_points_c, fp_losses = oulier_removal(fix_points_c, fp_losses_c )
        #fig1 = plot_fp_qualitiy(fp_losses, fp_losses_c)

        fp_r = []
        for f in fps:
            #print(f.shape, "fix_in shape")
            fp_readout, _ = get_read_out_projection(f, model)
            #print(fp_readout.shape, "fp readout shape")
            fp_r.append(fp_readout)
        r_r = []
        for r in runs:
            #print(r.shape, "run shape")
            r_readout = get_read_out_projection_run(r, model)
            print(r_readout.shape, "r_readout shape")
            r_r.append(r_readout)




        fp = np.concatenate(fps)
        print(f"fp shape {fp.shape}")
        fp_rs = np.concatenate(fp_r)
        print(f"fp_rs shape {fp_rs.shape}")
        rs = np.concatenate(runs)
        print(f"runs shape {rs.shape}")
        runs_readout = np.concatenate(r_r)
        print(f"frns_readout shape {runs_readout.shape}")

        #print(f"run readout")

        #fig3 = plot_fixpoints(fp, fp_rs)
        fig = plot_with_runs(fp, fp_r, rs, runs_readout)
        plt.show()



    if mode == 'leon':
        plot_with_runs('/home/falconinae/Documents/University/NDyn/RevRec/models/GRU_integration_07-08-2020_05-46-00_PM/fixpoints_07-08-2020_11-40-14_PM', '/home/falconinae/Documents/University/NDyn/RevRec/models/GRU_integration_07-08-2020_05-46-00_PM/exemplary_runs_09-08-2020_02-32-55_PM')
