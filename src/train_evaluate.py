"""
train and evaluate
"""

import os
import argparse
import logging
import numpy as np
import pickle
from scipy import sparse

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from data_utils import *
from utils import *
from data_load import load_data
from gaussian_data import *

from lssvm import LSSVM
import pandas as pd

SEED = 1021

categs_in_short = {'rest':'rest', 'balancedrest':'balrest', 'comp.sys.ibm.pc.hardware':'ibm', 'comp.sys.mac.hardware':'mac', 'talk.religion.misc':'relmisc',
                   'soc.religion.christian':'christ', 'misc.forsale':'4sale', 'talk.politics.guns':'polguns', 'talk.politics.misc':'polmisc',
                   'Health':'health', 'Sports':'sports', 'Business & Finance':'busfin', 'Society & Culture':'cult', 'Science & Mathematics':'scimath',
                   'Education & Reference':'edu', 'Family & Relationships':'family'}

def parse_args():
    parser = argparse.ArgumentParser('train-evaluate')
    parser.add_argument('-o', '--model_dir', default='../../experiments/', help="Directory to save results")
    parser.add_argument('-d', '--data_name', type=str, default='yqa')
    parser.add_argument('--data_path', type=str, default='../../data/')
    parser.add_argument('-f', '--model_name', type=str, default='lssvm')
    parser.add_argument('-e', '--emb_name', type=str, default='dc-tfidf-glv',
                        help='for tfidf use "dc-tfidf", for glove use "dc-tfidf-glv", for word2vec use "dc-tfidf-w2v')
    parser.add_argument('-c', '--categs', default=None, type=str, nargs='+')
    parser.add_argument('-n', '--N', default=None, type=int, help='number samples')
    parser.add_argument('--Ntest', default=None, type=int, help='Gaussian data test size, only used for gaussian data')
    parser.add_argument('-p', '--P', default=None, type=int, help='feature dimension, used for gaussian data')
    parser.add_argument('-k', '--kernel_type', default='rbf', type=str,
                        help='for RBF(Gaussian) kernel "rbf", for polynomial kernel "poly"')
    parser.add_argument('-s', '--sigma2', default=1, type=float, help="hyperparameter for rbf kernel")
    parser.add_argument('-t', '--derivs', default=None, type=float, nargs='+',
                        help="hyperparameter for polynomial kernel determines f'/f''")
    parser.add_argument('-x', '--x_stats', default=False, action="store_true",
                        help='Switch to enable to computation of x stats (True: x stats enabled)')
    parser.add_argument('-r', '--tau_ratio', default=None, type=float, help='used for the folder naming')
    parser.add_argument('-g', '--g_type', default='v1_4', type=str, help='gaussian data type')
    return parser.parse_args()


def train_model_and_evaluate(X_train, y_train, X_test, model_name, model_dir=None, plot_decision=True, data_name=None, y_test=None, params=None, means=None, covs=None):
    if data_name != 'gaussian':
        if sparse.issparse(X_train):
            X_train = X_train.toarray()
            X_test = X_test.toarray()
        # normalize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    if model_name == 'lssvm':
        X_train = X_train.T
        X_test = X_test.T
        if data_name != 'gaussian':
            y_train = (np.array(y_train) - 1/2) * 2
        logging.info('starting to train LS-SVM')
        clf = LSSVM(kernel_type=params.kernel_type, sigma2=params.sigma2, gamma=1, derivs=params.derivs, means=means, covs=covs)
        clf.train(X_train, y_train)
        logging.info('starting to predict')
        y_pred = clf.evaluate(X_test)
        if plot_decision:
            estimated_perf = clf.theoretical_performance_estimate()
            logging.info('estimated performance: {}'.format(estimated_perf.flatten()[0]))
            with open(os.path.join(model_dir, 'estimated_perf.txt'), 'w') as f:
                f.write('{}\n'.format(estimated_perf.flatten()[0]))
            logging.info('plotting decision histogram')
            if data_name != 'gaussian':
                y_test = (np.array(y_test) - 1 / 2) * 2
            fig = clf.plot_decision_g(X_test, y_test)
            fig.savefig(os.path.join(model_dir, 'hist_decision.pdf'))
    return y_pred


def evaluation(path_to_save, y_test, y_pred, target_names, correct_labels=False):
    if correct_labels:
        y_test = (np.array(y_test) - 1 / 2) * 2
    logging.info(classification_report(y_test, y_pred, target_names=target_names))
    metric_fns = [accuracy_score]
    metric_names = ['accuracy']
    scores = [m_fn(y_test, y_pred) for m_fn in metric_fns]
    with open(os.path.join(path_to_save, 'res.csv'), 'w') as writer:
        writer.write(','.join(metric_names) + '\n')
        writer.write(','.join(list(map(str, scores))) + '\n')
    fig = print_confusion_matrix(confusion_matrix(y_test, y_pred), target_names)
    fig.savefig(os.path.join(path_to_save, 'confusion_mtx.pdf'), bbox_inches='tight')


def write_x_stats(model_dir, X, y=None, data_name=None, sigma2=1):
    if sparse.issparse(X):
        X = X.toarray()
    if data_name != 'gaussian':
        X = normalize_bysubmean(X)
    #
    (mean, cov), (means, covs) = x_stats(X.T, [0.5, 0.5], y=y)
    pickle.dump({'means': means, 'covs': covs}, open(os.path.join(model_dir, 'x_stats.pkl'), 'wb'))
    write_stats(model_dir, mean=mean, cov=cov, means=means, covs=covs)
    logging.info('calculating intra class stats')
    means, stds = stats_for_inter_intra_class_differences(X, y=y)
    df_mstd = pd.DataFrame(zip(means, stds), columns=['means', 'stds'])
    df_mstd.to_csv(os.path.join(model_dir, 'inter_intra_class_differences_means_stds.csv'), sep=',', header=True)
    #
    p = X.shape[1]
    xij_dist = xij_distances(X, p)
    histogram_of_x(model_dir, xij_dist, step_size=0.0025)
    #
    logging.info('sigma2: {}'.format(sigma2))
    k_mtx = kernel_mtx(X, p, f='rbf', sigma2=sigma2)
    pickle.dump(k_mtx, open(os.path.join(model_dir, 'rbf_g{}_kernel_mtx.pkl'.format(sigma2)), 'wb'))
    plot_kernel_mtx(model_dir, k_mtx)
    #
    plot_write_eig_vecs_kernel_mtx(model_dir, k_mtx, outnameprefix='_g{}'.format(sigma2))


def main(args):
    # reset logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Set the logger
    d_name = args.data_name
    if args.data_name != 'gaussian':
        if args.tau_ratio:
            assert args.model_name == 'lssvm'
            m_name = '{}_{}'.format(args.model_name, args.kernel_type)
            args.model_dir = args.model_dir.format('{}_{}{}'.format(d_name, categs_in_short[args.categs[0]], categs_in_short[args.categs[1]]), args.emb_name, m_name, m_name, args.tau_ratio)
        else:
            if args.model_name == 'lssvm':
                m_name = '{}_{}'.format(args.model_name, args.kernel_type)
                if not args.sigma2 is None:
                    m_name = m_name + '_s{}'.format(args.sigma2)
                args.model_dir = args.model_dir.format('{}_{}{}'.format(d_name, categs_in_short[args.categs[0]], categs_in_short[args.categs[1]]), args.emb_name, m_name, m_name, args.N)
            else:
                args.model_dir = args.model_dir.format('{}_{}{}'.format(d_name, categs_in_short[args.categs[0]], categs_in_short[args.categs[1]]), args.emb_name, args.model_name, args.model_name, args.N)
    else:
        if args.model_name == 'lssvm':
            m_name = '{}_{}'.format(args.model_name, args.kernel_type)
        else:
            m_name = args.model_name
        if args.tau_ratio:
            assert args.model_name == 'lssvm'
            args.model_dir = args.model_dir.format(d_name, '{}_{}'.format(d_name, args.g_type), m_name, m_name, args.tau_ratio)
        else:
            args.model_dir = args.model_dir.format(d_name,  '{}_{}'.format(d_name, args.g_type), m_name, m_name, "{}_p{}".format(args.N, args.P))

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info('logging file under {}'.format(args.model_dir))
    logging.info('args: {}'.format(args.__dict__))
    #
    if 'w2v' in args.emb_name:
        emb_filename = os.path.join(args.data_path, 'w2v/GoogleNews-vectors-negative300.bin')
    elif 'glv' in args.emb_name:
        emb_filename = os.path.join(args.data_path, 'glv/glove.6B.300d.txt.word2vec') # https://radimrehurek.com/gensim/scripts/glove2word2vec.html
    else:
        emb_filename = None
    if args.data_name == 'gaussian':
        (X_train, y_train), (X_test, y_test), (means, covs) = gaussian_data_load(args.P, args.N, type=args.g_type, n_test=args.Ntest)
    else:
        (X_train, y_train), (X_test, y_test) = load_data(args.data_path, args.data_name, args.emb_name, categories=args.categs, train_sample_size=args.N, emb_filename=emb_filename)
        means, covs = None, None
    if args.model_name:
        logging.info('Starting to train')
        y_pred = train_model_and_evaluate(X_train, y_train, X_test, args.model_name, model_dir=args.model_dir, data_name=args.data_name, y_test=y_test, params=args, means=means, covs=covs)
        logging.info('Starting to train evaluate')
        evaluation(args.model_dir, y_test, y_pred, args.categs, correct_labels=((args.model_name == 'lssvm') and (args.data_name !='gaussian')))
    #
    if args.x_stats:
        logging.info('Starting to empirical analysis')
        write_x_stats(args.model_dir, X_train, y=y_train, data_name=args.data_name)


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(SEED)
    logging.info('args: {}'.format(args))
    main(args)