import numpy as np


def gaussian_data_generate(class_sizes=None, means=None, covariances=None):
    covs = covariances
    p = len(means[0])
    X_train = np.array([]).reshape(p, 0)
    y_train = []
    K = len(class_sizes)
    for i in range(K):
        X_train = np.concatenate((X_train, np.random.multivariate_normal(means[i], covs[i], size=np.int(class_sizes[i])).T),axis=1)
        y_train = np.concatenate((y_train, 2 * (i - K / 2 + .5) * np.ones(np.int(class_sizes[i]))))
    return X_train, y_train


def gaussian_data_load(p, n, type='v2_4', multipl_=2, n_test=None):
    if 'v0' == type:
        m_1 = np.zeros(p)
        m_2 = np.zeros(p)
        means = [m_1, m_2]
        covariances = [np.eye(p), np.eye(p)]
        class_sizes = [int(n / 2), int(n / 2)]
    elif 'v1' == type:
        m_1 = np.append(1, np.zeros(p - 1))
        m_2 = np.append(-1, np.zeros(p - 1))
        means = [multipl_ * m_1, multipl_ * m_2]
        covariances = [np.eye(p), np.eye(p)]
        class_sizes = [int(n / 2), int(n / 2)]
    elif 'v1_4' == type:
        multipl_ = 4
        m_1 = np.append(1, np.zeros(p - 1))
        m_2 = np.append(-1, np.zeros(p - 1))
        means = [multipl_ * m_1, multipl_ * m_2]
        covariances = [np.eye(p), np.eye(p)]
        class_sizes = [int(n / 2), int(n / 2)]
    elif 'v2' == type:
        m_1 = np.append(1, np.zeros(p-1))
        m_2 = np.append([0, 1], np.zeros(p-2))
        means = [multipl_ * m_1, multipl_ * m_2]
        covariances = [np.eye(p), toeplitz_fn(p)]
        class_sizes = [int(n / 2), int(n / 2)]
    elif 'v2_4' == type:
        multipl_ = 4
        m_1 = np.append(1, np.zeros(p-1))
        m_2 = np.append([0, 1], np.zeros(p-2))
        means = [multipl_ * m_1, multipl_ * m_2]
        covariances = [np.eye(p), toeplitz_fn(p)]
        class_sizes = [int(n / 2), int(n / 2)]
    elif 'v3_4' == type:
        multipl_ = 4
        m_1 = np.append(1, np.zeros(p - 1))
        means = [multipl_ * m_1, - multipl_ * m_1]
        covariances = [np.eye(p), toeplitz_fn(p)]
        class_sizes = [int(n / 2), int(n / 2)]
    X_train, y_train = gaussian_data_generate(class_sizes=class_sizes, means=means, covariances=covariances)
    if n_test:
        class_sizes = [int(n_test / 2), int(n_test / 2)]
    X_test, y_test = gaussian_data_generate(class_sizes=class_sizes, means=means, covariances=covariances)
    return (X_train.T, y_train), (X_test.T, y_test), (means, covariances)


def toeplitz_fn(p, base=0.4, multipl=4):
    f = lambda i, j: base ** abs(i-j) * (1 + multipl/np.sqrt(p))
    toeplitz_mtx = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            toeplitz_mtx[i, j] = f(i, j)
    return toeplitz_mtx


def means_covariances(X_train, prob=[0.5, 0.5]):
    p = X_train.shape[0]
    n = X_train.shape[1]
    k = len(prob)
    index = []
    means = []
    covs = []
    tmp = 0
    for i in range(k):
        index.append(np.arange(tmp, tmp + int(n * prob[i]), 1))
        means.append(np.mean(X_train[:, index[i]], axis=1).reshape(p, 1))
        covs.append(np.cov(X_train[:, index[i]]))
        tmp = tmp + int(n * prob[i]) - 1
    return (means, covs)