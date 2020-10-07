import pickle
import logging
import pandas as pd
from data_utils import *
from sklearn.datasets import fetch_20newsgroups
import os
from scipy import sparse



def load_data(data_path, data_name, emb_name, categories=None, train_sample_size=None, emb_filename=None):
    logging.info('Loading data')
    if data_name == 'yqa':  # Yahoo QA
        path_to_data_files = os.path.join(data_path, 'yahoo_answers_csv')
    elif data_name == 'n20':  # 20 News
        path_to_data_files = os.path.join(data_path, '20news')
    X_train, y_train, X_test, y_test = data_feature_vecs_labels(path_to_data_files, data_name, emb_name, categories=categories, train_sample_size=train_sample_size, emb_filename=emb_filename)
    return (X_train, y_train), (X_test, y_test)


def read_yahoo_csv(csv_file):
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        x, y = [], []
        for line in lines:
            list_line = line.replace('"', '').replace('\n', '').split(',')
            if len(list_line) < 2:
                print(line)
            y.append(int(list_line[0]))
            x.append(" ".join(list_line[1:]))
    return (x, y)


def load_yahoo_data(path_to_data_files, categories=['Health', 'Sports'], train_sample_size=None):
    text_train, y_train = read_yahoo_csv(os.path.join(path_to_data_files, 'train.csv'))
    text_test, y_test = read_yahoo_csv(os.path.join(path_to_data_files, 'test.csv'))
    label_names_txt = os.path.join(path_to_data_files, 'classes.txt')
    with open(label_names_txt, 'r') as f:
        lines = f.readlines()
        target_names = [line.replace('\n', '') for line in lines]
    # filter data and keep only classes in categories
    labels_of_chosen_categories = [target_names.index(t_n)+1 for t_n in target_names if t_n in categories] # labels from 1 to 10
    if not set(target_names) == set(categories):
        logging.info('labels of the {} categories : {}'.format(categories, labels_of_chosen_categories))
        train_df = pd.DataFrame(list(zip(text_train, y_train)), columns=['x', 'y'])
        train_df = train_df[train_df['y'].isin(labels_of_chosen_categories)]
        train_df = train_df.reset_index(drop=True)
        test_df = pd.DataFrame(list(zip(text_test, y_test)), columns=['x', 'y'])
        test_df = test_df[test_df['y'].isin(labels_of_chosen_categories)]
        # change labels that starting from 0 ..
        for idx, cat_i in enumerate(labels_of_chosen_categories):
            train_df = train_df.replace({'y': cat_i}, idx)
            test_df = test_df.replace({'y': cat_i}, idx)
    #
    (list_x_train, list_y_train), (list_x_test, list_y_test) = subsample_train_data(train_df, test_df, target_names, categories, train_sample_size)
    return (list_x_train, list_y_train), (list_x_test, list_y_test)


def load_20news_data(categories=['soc.religion.christian', 'misc.forsale'], train_sample_size=None):
    remove = ()
    if categories and ('rest' in categories or 'balancedrest' in categories):
        cats = None  # fetch all the class documents
    else:
        cats = categories
    data_train = fetch_20newsgroups(subset='train', categories=cats, shuffle=True, random_state=42, remove=remove)
    data_test = fetch_20newsgroups(subset='test', categories=cats, shuffle=True, random_state=42, remove=remove)
    y_train, y_test = data_train.target, data_test.target
    text_train, text_test = data_train.data, data_test.data
    target_names = data_train.target_names
    #
    train_df = pd.DataFrame(list(zip(text_train, y_train)), columns=['x', 'y'])
    test_df = pd.DataFrame(list(zip(text_test, y_test)), columns=['x', 'y'])
    (list_x_train, list_y_train), (list_x_test, list_y_test) = subsample_train_data(train_df, test_df, target_names, categories, train_sample_size)
    return (list_x_train, list_y_train), (list_x_test, list_y_test)


def data_feature_vecs_labels(path_to_data_files, data_name, emb_name, categories=None, train_sample_size=None, emb_filename=None):
    features_pkl_file = os.path.join(path_to_data_files, '{}_{}_n{}.pkl'.format('_'.join(categories), emb_name, train_sample_size))
    if os.path.exists(features_pkl_file):
        data_dict = pickle.load(open(features_pkl_file, 'rb'))
    else:
        if 'dc-tfidf' in emb_name:
            data_dict = dc_tfidf_data_dict(path_to_data_files, data_name, categories, train_sample_size)
            if ('w2v' in emb_name) or ('glv' in emb_name):
                X_tfidf_train, y_train, X_tfidf_test, y_test = data_dict['data']
                vocab_tfidf = data_dict['vocab']
                w2v_mtx = w2v_feature_mtx(emb_filename, vocab_tfidf)
                # weighted average of word embeddings in which weights are tf-idf weights
                X_train = sparse.csr_matrix.dot(X_tfidf_train, w2v_mtx)
                X_test = sparse.csr_matrix.dot(X_tfidf_test, w2v_mtx)
                data_dict['data'] = (X_train, y_train, X_test, y_test)
                logging.info('tf-idf w2v features are ready.')
        logging.info('Saving data dictionary.')
        pickle.dump(data_dict, open(features_pkl_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return data_dict['data']


def categ_fname(categ):
    new_categ = categ.replace(' & ', '')
    return new_categ

def dc_tfidf_data_dict(path_to_data_files, data_name, categories, train_sample_size):
    tfidf_pkl_file = os.path.join(path_to_data_files, '{}_{}_n{}.pkl'.format('_'.join(categories), 'dc-tfidf', train_sample_size))
    logging.info('trying loading tfidf pkl: {}'.format(tfidf_pkl_file))
    if os.path.exists(tfidf_pkl_file):
        logging.info('trying loading tfidf pkl: {}'.format(tfidf_pkl_file))
        data_dict = pickle.load(open(tfidf_pkl_file, 'rb'))
    else:
        logging.info('Loading data')
        if data_name == 'yqa':  # Yahoo QA
            (text_train, y_train), (text_test, y_test) = load_yahoo_data(path_to_data_files, categories=categories, train_sample_size=train_sample_size)
        elif data_name == 'n20':  # 20News
            (text_train, y_train), (text_test, y_test) = load_20news_data(categories=categories, train_sample_size=train_sample_size)
        X_train, X_test, vocab = dc_tfidf_features(text_train, text_test)
        data_dict = {'data': (X_train, y_train, X_test, y_test), 'vocab': vocab}
        pickle.dump(data_dict, open(tfidf_pkl_file, 'wb'))
    return data_dict




