import logging
import numpy as np
import string
import re
from spacy.lang.en import English
import spacy

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, cdist, squareform

from gensim.models import KeyedVectors
import pandas as pd

from utils import *


SEED = 1021

def remove_specialchars(text):
    pattern = r'[^a-zA-Z\s]' # r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, ' ', text)
    return re.sub(r'[^\w]', ' ', text)


def spacy_tokenizer(sentence):
    """
    https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/
    """
    # Create list of punctuation marks
    punctuations = string.punctuation

    # Create list of stopwords
    stop_words = spacy.lang.en.stop_words.STOP_WORDS

    # Load English tokenizer, tagger, parser, NER and word vectors
    parser = English()
    sentence = remove_specialchars(sentence)

    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

    # Removing stop words
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

    # return preprocessed list of tokens
    return mytokens


def dc_tfidf_features(text_train, text_test, max_features=3500):
    logging.info("Extracting features from the training dataset using a sparse vectorizer")
    vectorizer = TfidfVectorizer(max_features=max_features, tokenizer=spacy_tokenizer, use_idf=True)
    X_train = vectorizer.fit_transform(text_train)
    logging.info("X_train, n_samples: %d, n_features: %d" % X_train.shape)
    vocab = vectorizer.get_feature_names()
    if len(vocab) != max_features:
        logging.info('n_features ({}) smaller than the dim ({}) specified:'.format(len(vocab), max_features))
    logging.info("Extracting features from the test data using the same vectorizer")
    X_test = vectorizer.transform(text_test)
    logging.info("X_test, n_samples: %d, n_features: %d" % X_test.shape)
    return X_train, X_test, vocab


def wc_tfidf_features(text_train, max_features=3500, n=None):
    logging.info("Extracting features from the training dataset using a sparse vectorizer")
    vectorizer = TfidfVectorizer(max_features=max_features, tokenizer=spacy_tokenizer, use_idf=True)
    X_train = vectorizer.fit_transform(text_train)
    logging.info("X_train, n_samples: %d, n_features: %d" % X_train.shape)
    feature_names = vectorizer.get_feature_names()
    if len(feature_names) != max_features:
        logging.info('n_features ({}) smaller than the dim ({}) specified:'.format(len(feature_names), max_features))
    if n:
        words_tfidf_sum = np.sum(X_train.toarray(), axis=0)
        counts = np.count_nonzero(X_train.toarray(), axis=0)
        average_tf_idf_per_word = words_tfidf_sum / counts
        vocab = [feature_names[idx] for idx in np.argsort(-average_tf_idf_per_word)[:n]]
    else:
        vocab = feature_names
    return X_train, vocab


def w2v_feature_mtx(emb_filename, vocab_tfidf):
    logging.info('Loading w2v model')
    if 'bin' in emb_filename:
        model = KeyedVectors.load_word2vec_format(emb_filename, binary=True)
    else:
        model = KeyedVectors.load_word2vec_format(emb_filename, binary=False)
    w2v_dim = 300
    w2v_mtx = np.zeros((len(vocab_tfidf), w2v_dim))
    wc = 0
    for idx, v_i in enumerate(vocab_tfidf):
        if v_i in model:
            w2v_mtx[idx] = model[v_i]
            wc +=1
    logging.info('w2v matrix is ready. {} words appears in the pre-trained embeddings.'.format(wc))
    return w2v_mtx


def check_emb_vocabulary(tfidf_vocab, emb_filename):
    logging.info('Loading embbeding model')
    if 'bin' in emb_filename:
        model = KeyedVectors.load_word2vec_format(emb_filename, binary=True)
    else:
        model = KeyedVectors.load_word2vec_format(emb_filename, binary=False)
    logging.info('Restricting tfidf vocab by word embeddings')
    emb_vocab = list(model.wv.vocab)
    filtered_tfidf_vocab = [w for w in tfidf_vocab if w in emb_vocab]
    logging.info('{} words after filtering'.format(len(filtered_tfidf_vocab)))
    return filtered_tfidf_vocab


def kernel_mtx(X, p, metric='euclidean', f='squared_diff', gamma=None, sigma2=None):
    if f == 'squared_diff':
        pairwise_dists = squareform(pdist(X, metric=metric))
        square_distance = pairwise_dists ** 2
        K = (1 / p) * square_distance
    elif f == 'rbf':
        pairwise_dists = squareform(pdist(X, metric=metric))
        square_distance = (1 / p) * pairwise_dists ** 2
        if gamma:
            K = np.exp(-square_distance / gamma)
        elif sigma2:
            K = np.exp(-square_distance / (2*sigma2))
    return K


def xij_distances(X, p, metric='euclidean'):
    pairwise_dists = pdist(X, metric=metric)
    distance_div_by_sqrt_p = pairwise_dists / np.sqrt(p)
    return distance_div_by_sqrt_p


def x_stats(X_train, prob=[0.5, 0.5], y=None, label_bool=False):
    """mainly taken from https://github.com/Zhenyu-LIAO/RMT4LSSVM/blob/master/RMT4LSSVM.ipynb"""
    p = X_train.shape[0]
    n = X_train.shape[1]
    k = len(prob)
    index = []
    means = []
    covs = []
    tmp = 0
    for i in range(k):
        if label_bool:
            index.append(np.where(y==i)[0])
        else:
            index.append(np.arange(tmp, tmp + int(n * prob[i]), 1))
        means.append(np.mean(X_train[:, index[i]], axis=1).reshape(p, 1))
        covs.append(np.cov(X_train[:, index[i]]))
        tmp = tmp + int(n * prob[i]) - 1
    mean = np.mean(X_train, axis=1).reshape(p, 1)
    cov = (X_train @ X_train.T - mean @ mean.T).reshape(p, p)
    return (mean, cov), (means, covs)


def subsample_train_data(train_df, test_df, target_names, categories, train_sample_size):
    # sample N instances from training
    labels = sorted(list(train_df['y'].unique()))
    logging.info('nr of docs in {}: {}'.format(categories, len(train_df)))
    if train_sample_size:
        list_x_train, list_y_train = [], []
        np.random.seed(SEED)
        if categories and ('balancedrest' in categories):
            class_name_one = categories[0]
            label_one = target_names.index(class_name_one)
            df_train = train_df[train_df['y'] == label_one]
            if df_train.shape[0] < train_sample_size / 2.:
                train_rest_sample_size = train_sample_size - df_train.shape[0]
                list_x_train.extend(df_train['x'].tolist())
                list_y_train.extend(df_train['y'].tolist())
            else:
                train_rest_sample_size = train_sample_size - int(train_sample_size / 2)
                sample_indices = np.random.choice(len(df_train), size=int(train_sample_size / 2), replace=False)
                df_train = df_train.iloc[sample_indices]
                list_x_train.extend(df_train['x'].tolist())
                list_y_train.extend(df_train['y'].tolist())
            logging.info('# of sample_indices: {}'.format(len(list_y_train)))
            rest_labels = [l_i for l_i in labels if l_i != label_one]
            for l_i in rest_labels[:-1]:
                df_train = train_df[train_df['y'] == l_i]
                df_train = df_train.reset_index(drop=True)
                sample_indices = np.random.choice(len(df_train),
                                                  size=int(np.floor(train_rest_sample_size / len(rest_labels))),
                                                  replace=False)
                logging.info('# of sample_indices: {}'.format(len(sample_indices)))
                df_train = df_train.iloc[sample_indices]
                list_x_train.extend(df_train['x'].tolist())
                list_y_train.extend(df_train['y'].tolist())
            # to get exact sample size; complete with last one
            l_i = rest_labels[-1]
            nr_rest_to_sample = train_sample_size - len(list_y_train)
            logging.info('len(list_y_train): {}'.format(len(list_y_train)))
            logging.info('nr_rest_to_sample: {}'.format(nr_rest_to_sample))
            df_train = train_df[train_df['y'] == l_i]
            df_train = df_train.reset_index(drop=True)
            sample_indices = np.random.choice(len(df_train), size=nr_rest_to_sample, replace=False)
            logging.info('# of sample_indices: {}'.format(len(sample_indices)))
            df_train = df_train.iloc[sample_indices]
            list_x_train.extend(df_train['x'].tolist())
            list_y_train.extend(df_train['y'].tolist())
        else:
            for l_i in labels:
                df_train = train_df[train_df['y'] == l_i]
                df_train = df_train.reset_index(drop=True)
                sample_indices = np.random.choice(len(df_train), size=int(train_sample_size / len(labels)),
                                                  replace=False)
                logging.info('# of sample_indices: {}'.format(len(sample_indices)))
                df_train = df_train.iloc[sample_indices]
                list_x_train.extend(df_train['x'].tolist())
                list_y_train.extend(df_train['y'].tolist())
    else:
        list_x_train = train_df['x'].tolist()
        list_y_train = train_df['y'].tolist()
    #
    if categories and ('rest' in categories or 'balancedrest' in categories):
        train_df = pd.DataFrame(list(zip(list_x_train, list_y_train)), columns=['x', 'y'])
        # assign new labels to have one (1) versus rest (0)
        class_name_one = categories[0]
        assert class_name_one != 'rest'
        label_one = target_names.index(class_name_one)  # the label to be changed to 0
        label_map = {label_i: 1 for label_i in labels}  # dictionary contains the label mapping to one vs. rest
        label_map[label_one] = 0
        #
        for orig_label, new_label in label_map.items():
            train_df = train_df.replace(orig_label, new_label)
            test_df = test_df.replace(orig_label, new_label)
        list_y_train = train_df['y'].tolist()
    list_x_test = test_df['x'].tolist()
    list_y_test = test_df['y'].tolist()
    return (list_x_train, list_y_train), (list_x_test, list_y_test)


def normalize_bysubmean(X):
    logging.info('normalizing X, X.shape:{}'.format(X.shape))
    n, p = X.shape
    X_t = X.T - np.mean(X, axis=0).reshape(p, 1)
    X_normed = X_t.T * np.sqrt(p) / np.sqrt(np.sum(X_t ** 2) / n)
    return X_normed


def stats_for_intraclass_differences(X, probs=[0.5, 0.5], y=None):
    assert probs == [0.5, 0.5]  # TODO: extend to different proportions and multiclass
    n, p = X.shape
    assert n % 2 == 0
    X1 = X[:int(n / 2), :]
    X2 = X[int(n / 2):, :]
    X1_ij = xij_distances(X1, p)
    X2_ij = xij_distances(X2, p)
    X_ij = np.append(X1_ij, X2_ij)
    return np.mean(X_ij), np.std(X_ij)


def stats_for_interclass_differences(X, probs=[0.5, 0.5], y=None):
    assert probs == [0.5, 0.5]  # TODO: extend to different proportions and multiclass
    n, p = X.shape
    assert n % 2 == 0
    X1 = X[:int(n / 2), :]
    X2 = X[int(n / 2):, :]
    xij_dists = np.zeros(len(X1)*len(X2))
    idx = 0
    for i in range(len(X1)):
        for j in range(len(X2)):
            x_i = X1[i]
            x_j = X2[j]
            dist_ij = np.linalg.norm(x_i - x_j)
            xij_dists[idx] = dist_ij
            idx += 1
    X_ij = np.array(xij_dists) / np.sqrt(p)
    return np.mean(X_ij), np.std(X_ij)


def stats_for_inter_intra_class_differences(X, probs=[0.5, 0.5], y=None):
    mean_inter, std_inter = stats_for_interclass_differences(X, probs=probs, y=y)
    mean_intra, std_intra = stats_for_intraclass_differences(X, probs=probs, y=y)
    return (mean_inter, mean_intra), (std_inter, std_intra)



