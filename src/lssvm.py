from scipy.spatial.distance import pdist, cdist, squareform
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import special as sp

class LSSVM:
    """some parts are taken from https://github.com/Zhenyu-LIAO/RMT4LSSVM/blob/master/RMT4LSSVM.ipynb"""
    def __init__(self, kernel_type=None, gamma=1, sigma2=1, metric='euclidean', derivs=[3, -.5, 2], means=None, covs=None):
        self.sigma2 = sigma2 # rbf kernel sigma_square
        self.gamma = gamma
        self.metric = metric # euclidean !
        self.kernel_type = kernel_type
        self.derivs = derivs
        self.means = means
        self.covs = covs
        if kernel_type == 'rbf':
            self.kernel_fn = self.rbf_kernel
        if kernel_type == 'poly':
            self.kernel_fn = self.polynomial_kernel
        elif kernel_type == 'sdp':
            self.kernel_fn = self.squared_distance_kernel
        elif kernel_type == 'linear':
            self.kernel_fn = self.linear_kernel

    def rbf_kernel(self, X, x=None, eval=False):
        p = X.shape[0]
        if eval:
            square_distance = (1 / p) * (np.expand_dims(np.diag(X.T @ X), axis=1) + np.expand_dims(np.diag(x.T @ x), axis=1).T - 2 * X.T @ x)
        else:
            pairwise_dists = squareform(pdist(X.T, metric=self.metric))
            square_distance = (1 / p) * pairwise_dists ** 2
        K = np.exp(-square_distance / (2 * self.sigma2))
        return K

    def squared_distance_kernel(self, X, x=None, eval=False):
        p = X.shape[0]
        if eval:
            n = X.shape[1]
            Xx_dists = X - np.tile(x, (n, 1))
            square_distance = Xx_dists ** 2
        else:
            pairwise_dists = squareform(pdist(X.T, metric=self.metric))
            square_distance = pairwise_dists ** 2
        K = (1 / p) * square_distance
        return K

    def derivs_to_coeffs(self, derivs):
        coeffs = np.zeros(3)
        for i in range(3):
            coeffs[i] = derivs[2 - i] / np.math.factorial(2 - i)
        return coeffs

    def polynomial_kernel(self, X, x=None, eval=False):
        # computation of tau
        p, n = X.shape
        XX = X.T @ X / p # .shape: (n,n)
        tau = 2*np.trace(XX) / n
        # polynomial function
        coeffs = self.derivs_to_coeffs(self.derivs)
        if eval:
            square_distance = (1 / p) * ( np.expand_dims(np.diag(X.T @ X), axis=1) + np.expand_dims(np.diag(x.T @ x), axis=1).T - 2 * X.T @ x)
        else:
            pairwise_dists = squareform(pdist(X.T, metric=self.metric))
            square_distance = (1 / p) * pairwise_dists ** 2
        K = np.polyval(coeffs, (square_distance - tau))
        return K


    def linear_kernel(self, X, x=None):
        if x:
            K = np.dot(X.T, x)
        else:
            K = np.dot(X.T, X)
        return K

    def build_decision_boundary(self, K, y):
        n = K.shape[0]
        self.S_inv = np.linalg.inv(K + (n/self.gamma)*np.eye(n))
        self.b = np.sum(self.S_inv @ y) / np.sum(self.S_inv)
        self.alpha = self.S_inv @ (y - self.b)
        # S = K + n / self.gamma * np.eye(n)
        # invS_y = scipy.linalg.solve(S, y)
        # invS_1 = scipy.linalg.solve(S, np.ones(n))
        # self.b = invS_y.sum() / invS_1.sum()
        # self.alpha = invS_y - invS_1 * self.b

    def train(self, X, y):
        self.X = X
        self.y = y
        K = self.kernel_fn(X)
        self.build_decision_boundary(K, y)

    def decision_function(self, x):
        K_x = self.kernel_fn(self.X, x, eval=True)
        g = self.alpha.T @ K_x + self.b

        return g

    def evaluate(self, x):
        self.x_test = x
        g = self.decision_function(x)
        y_pred = np.sign(g)
        return y_pred

    def plot_decision_g(self, x, y_test, step_size=None, n_bins=100):
        g = self.decision_function(x)
        g_1 = g[np.where(y_test == 1)]
        g_2 = g[np.where(y_test == -1.0)]
        fig = plt.figure()
        if step_size:
            n_digits = {0.0001:4, 0.001: 3, 0.01: 2, 0.1: 1}[step_size]
            bins1 = np.arange(round(min(g_1), n_digits), round(max(g_1), n_digits), step_size)
            bins2 = np.arange(round(min(g_2), n_digits), round(max(g_2), n_digits), step_size)

        else:
            bins1 = np.arange(min(g_1), max(g_1), n_bins)
            bins2 = np.arange(min(g_2), max(g_2), n_bins)
        n, bins, patches = plt.hist(g_1, bins=bins1, density=True, facecolor='blue')
        n, bins, patches = plt.hist(g_2, bins=bins2, density=True, facecolor='red')
        var_th, E_a = self.theoretical_decision_g()
        #
        xs = np.linspace(min(g), max(g), 100)
        g_th1 = scipy.stats.norm.pdf(xs, loc=E_a[0], scale=np.sqrt(max(var_th[0], 1e-5))).reshape(100, 1)
        g_th2 = scipy.stats.norm.pdf(xs, loc=E_a[1], scale=np.sqrt(max(var_th[1], 1e-5))).reshape(100, 1)
        pl1, = plt.plot(xs, g_th1, 'green')
        pl2, = plt.plot(xs, g_th2, 'purple')
        return fig

    def theoretical_decision_g(self, prob=[0.5, 0.5], correct=True):
        X_train = self.X
        p, n = X_train.shape
        if self.means:
            means, covs = self.means, self.covs
            correct = False
        else:
            means, covs = self.means_covariances(X_train)
        # computation of tau
        XX_train = X_train.T @ X_train / p
        tau = 2 * np.trace(XX_train) / n
        if self.kernel_type == 'rbf':
            f_tau = np.exp(-tau/(2 * self.sigma2))
            derivs = [f_tau, -f_tau/(2*self.sigma2), f_tau/(4*self.sigma2**2)]
        elif self.kernel_type == 'poly':
            derivs = self.derivs

        t1 = np.trace(covs[0] - prob[0] * covs[0] - prob[1] * covs[1]) / np.sqrt(p)
        t2 = np.trace(covs[1] - prob[0] * covs[0] - prob[1] * covs[1]) / np.sqrt(p)
        D = -2 * derivs[1] * (np.linalg.norm(means[1] - means[0])) ** 2 / p + derivs[2] * (t1 - t2) ** 2 / p + 2 * \
            derivs[2] * (np.trace((covs[0] - covs[1]) @ (covs[0] - covs[1]))) / (p ** 2)
        if correct:
            # D = D - 2 * derivs[2] * (np.trace(covs[0] ** 2)) / (n * p) - 2 * derivs[2] * np.trace(covs[1] ** 2) / (n * p)
            D = D - 2 * derivs[2] * (np.trace(covs[0]) ** 2) / (p * p * n * prob[0]) - 2 * derivs[2] * (np.trace(covs[1]) ** 2) / (p * p * n * prob[1])
        E_a = (prob[1] - prob[0]) * np.array([1.0, 1.0]) + 2 * prob[0] * prob[1] * self.gamma * D * np.array([-prob[1], prob[0]])

        V11 = (t2 - t1) ** 2 * derivs[2] ** 2 * np.trace(covs[0] @ covs[0]) / (p ** 3)
        V12 = (t2 - t1) ** 2 * derivs[2] ** 2 * np.trace(covs[1] @ covs[1]) / (p ** 3)
        V21 = 2 * derivs[1] ** 2 * (means[1] - means[0]).T @ covs[0] @ (means[1] - means[0]) / (p ** 2)
        V22 = 2 * derivs[1] ** 2 * (means[1] - means[0]).T @ covs[1] @ (means[1] - means[0]) / (p ** 2)
        V31 = 2 * derivs[1] ** 2 * (np.trace(covs[0] @ covs[0]) / prob[0] + np.trace(covs[0] @ covs[1]) / prob[1]) / ( n * p ** 2)
        V32 = 2 * derivs[1] ** 2 * (np.trace(covs[0] @ covs[1]) / prob[0] + np.trace(covs[1] @ covs[1]) / prob[1]) / ( n * p ** 2)
        if correct:
            V11 = V11 - (t2 - t1) ** 2 * derivs[2] ** 2 * (np.trace(covs[0]) ** 2) / (n * prob[0] * p**3)
            V12 = V12 - (t2 - t1) ** 2 * derivs[2] ** 2 * (np.trace(covs[1]) ** 2) / (n * prob[1] * p**3)
            V31 = V31 - 2 * derivs[1] ** 2 * ((np.trace(covs[0]) ** 2) / prob[0]) / (n * n * prob[0] * p ** 2)
            V32 = V32 - 2 * derivs[1] ** 2 * ((np.trace(covs[1]) ** 2) / prob[1]) / (n * n * prob[1] * p ** 2)

        var_th = 8 * self.gamma ** 2 * (prob[0] * prob[1]) ** 2 * np.array([V11 + V21 + V31, V12 + V22 + V32])
        return var_th, E_a
    
    def means_covariances(self, X_train, prob=[0.5, 0.5]):
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

    def qfunc(self, arg):
        return 0.5 - 0.5 * sp.erf(arg / 1.414)

    def theoretical_performance_estimate(self):
        var_th, E_a = self.theoretical_decision_g()
        error = self.qfunc((E_a[1] - E_a[0]) / (2 * np.sqrt(var_th[0]))) * 1 / 2 + (1 - self.qfunc((E_a[0] - E_a[1]) / (2 * np.sqrt(var_th[1])))) * 1 / 2
        return 1 - error

