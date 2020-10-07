import os
import logging
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    taken from

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the outputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def write_stats(model_dir, mean=None, cov=None, means=None, covs=None):
    mu_diff_norm = np.linalg.norm(means[1] - means[0])
    tr_diff_cov = np.trace(covs[0] - covs[1])
    # tr_diff_cov_square = np.trace((covs[0] - covs[1])**2)
    tr_diff_cov_square = np.trace((covs[0] - covs[1])@(covs[0] - covs[1]))
    tr_cov = np.trace(cov)
    tr_cov_square = np.trace(cov@cov)
    with open(os.path.join(model_dir, 'x_stats.csv'), 'w') as f:
        f.write('tr(C),tr(C^2),mean_diff_norm,tr(C2-C1),tr((C2-C1)^2)\n')
        f.write('{},{},{},{},{}\n'.format(tr_cov, tr_cov_square, mu_diff_norm, tr_diff_cov, tr_diff_cov_square))


def histogram_of_x(model_dir, xij_dist, step_size=0.01, fname='hist_diff_xij', save_txt=True, ax=None):
    n_digits = {0.001:3, 0.0025:3, 0.005:3, 0.01:2, 0.05:2, 0.1:1}[step_size]
    if np.isreal(xij_dist).all():
        xij_dist = np.real_if_close(xij_dist)
    else:
        logging.info('complex number encountered!')
    bbins = np.arange(round(min(xij_dist), n_digits), round(max(xij_dist), n_digits), step_size)
    #
    if ax:
        n, bins, patches = ax.hist(xij_dist, bins=bbins, density=True)
    else:
        plt.figure()
        n, bins, patches = plt.hist(xij_dist, bins=bbins, density=True)
        plt.savefig(os.path.join(model_dir, '{}.pdf'.format(fname)))
    if save_txt:
        df = pd.DataFrame([(str(round(b_i, n_digits)).ljust(n_digits+2, '0'), str(n_i)) for (b_i, n_i) in zip(bins, n)], columns=['x', 'y'])
        df.to_csv(os.path.join(model_dir, '{}.txt'.format(fname)), sep=' ', header=False, index=False)


def plot_write_eig_vecs_kernel_mtx(model_dir, k_mtx, outnameprefix=''):
    # eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(k_mtx)
    logging.info("max eigen val {}".format(vals.max()))
    logging.info("min eigen val {}".format(vals.min()))
    #
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 7))
    for idx, ax_i in enumerate(axes.flatten()):
        v_i = pd.Series(vecs[:, idx])
        ax_i.plot(v_i)
        ax_i.title.set_text('V{} eigenval {}'.format(idx + 1, str(round(vals[idx].real, 2))))
        v_i.to_csv(os.path.join(model_dir, 'k_mtx{}_eig_vector{}.txt'.format(outnameprefix, idx)), sep=' ', header=False)
        means, stds = stats_for_intraclass(np.array(v_i.tolist()).reshape(len(v_i.tolist()), 1))
        df_mstd = pd.DataFrame(zip(means,stds), columns =['means', 'stds'])
        df_mstd.to_csv(os.path.join(model_dir, 'k_mtx{}_eig_vector{}_means_stds.txt'.format(outnameprefix, idx + 1)), sep=',', header=True)
    plt.savefig(os.path.join(model_dir, 'k_mtx{}_eig_vectors.pdf'.format(outnameprefix)), bbox_inches='tight')
    #
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 7))
    for idx, ax_i in enumerate(axes.flatten()):
        v_i = pd.Series(vecs[:, idx])
        histogram_of_x(model_dir, v_i.to_list(), fname='v{}_hist'.format(idx + 1), save_txt=True, ax=ax_i, step_size=0.0025)
    plt.savefig(os.path.join(model_dir, 'k_mtx{}_eig_vector_hists.pdf'.format(outnameprefix)), bbox_inches='tight')


def plot_kernel_mtx(model_dir, k_mtx):
    fig, ax = plt.subplots(ncols=1, figsize=(4, 3))  # Sample figsize in inches
    ax = sns.heatmap(k_mtx)
    ax.xaxis.tick_bottom()
    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, 'kernel_mtx.pdf'), bbox_inches='tight')


def stats_for_intraclass(X, probs=[0.5,0.5]):
    logging.info('checking intraclass probs')
    assert probs == [0.5, 0.5] # TODO: extend to different proportions and multiclass
    n, p = X.shape
    assert n %2 == 0
    if n ==1 or p ==1:
        if np.isreal(X).all():
            X = np.real_if_close(X)
    X1 = X[:int(n/2), :]
    X2 = X[int(n/2):, :]
    xs = [X1, X2]
    means = [np.mean(x_i) for x_i in xs]
    stds = [np.std(x_i) for x_i in xs]
    return means, stds
