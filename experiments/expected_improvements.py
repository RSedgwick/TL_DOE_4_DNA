import copy
import numpy as np
from scipy.stats import ncx2, norm


class ExpectedImprovement:
    """This expected improvement class creates an acquisition function based on the target vector optimisation, except
    the objective values are scaled by constants alpha. This means we can alter which objectives are most important. If
    alphas = [1]*len(params) then we recover the original target vector optimisation"""

    def __init__(self, params, alphas=None):

        """:param params: list of parameters. should be either or both of 'm' and 'r'
        :param alphas: list of floats. If None, then alphas = [1]*len(params)"""

        if alphas is None:
            self.alphas = {param: 1 for param in params}
        else:
            self.alphas = alphas

    def filter_ys(self, ys, targets, params):
        """filter ys so all drifts (m) below the target drift are equal to the target
        :param ys: candas parameter array of the data
        :param targets: candas parameter array of the targets
        :param params: list of parameters. should be either or both of 'm' and 'r'
        :return: candas parameter array of the data with the drifts (m) below the target drift equal to the target
        """
        ys = copy.deepcopy(ys)
        if 'm' in params:
            targ_m = targets['m'].values()
            ms = ys['m'].values()
            ms[ms < targ_m] = targ_m
            ys['m'] = ms

        return ys

    def Chi_EI(self, mu, sig2, target, best_yet, k=1):
        """Expected improvement function taken from https://github.com/akuhren/target_vector_estimation
        :param mu: array of means
        :param sig2: array of variances
        :param target: array of targets
        :param best_yet: float of the best value so far
        :param k: int of the number of objectives
        :return: array of the expected improvement"""

        gamma2 = sig2.mean(axis=1)

        nc = ((target - mu) ** 2).sum(axis=1) / gamma2

        h1_nx = ncx2.cdf((best_yet / gamma2), k, nc)
        h2_nx = ncx2.cdf((best_yet / gamma2), (k + 2), nc)
        h3_nx = ncx2.cdf((best_yet / gamma2), (k + 4), nc)

        t1 = best_yet * h1_nx
        t2 = gamma2 * (k * h2_nx + nc * h3_nx)

        return t1 - t2

    def BestYet(self, ys, target_parrays):
        """Function to calculate the closest distance from the target we have observed so far
        :param ys: a series containing the values in the train set
        :param target_parrays: a 2x1 array containing the desired values for the rate and drift parameters
        :return best: the smallest distance between an observation and the target, a float"""

        ys_ = np.array([self.alphas[param] * ys[param] for param in ys.keys()])
        target = np.hstack(
            [self.alphas[param] * np.atleast_2d(target_parrays[param].z.values()) for param in ys.keys()])

        assert (ys_.shape[0] == target.T.shape[0])
        best = ((ys_ - target.T) ** 2).sum(axis=0).min()
        return best

    def EI(self, preds, target_parrays, best_yet, params):
        """Expected improvement using target vector optimisation
        :param preds: a candas uncertain parameter array containing the mean and variance of the predictions
        :param target_parrays: a 2x1 array containing the desired values for the rate and drift parameters
        :param best_yet: the smallest distance between an observation and the target, a float
        :param params: list of parameters. should be either or both of 'm' and 'r'
        :return ei: the expected improvements of each prediction"""

        mu = np.array([self.alphas[param] * preds[f'{param}_mu_z'].ravel() for param in params]).T
        sig2 = np.array([self.alphas[param] ** 2 * preds[f'{param}_sig2_z'].ravel() for param in params]).T
        target = np.hstack([self.alphas[param] * np.atleast_2d(target_parrays[param].z.values()) for param in params])

        k = len(params)
        ei = self.Chi_EI(mu, sig2, target, best_yet, k=k)

        return ei

    def get_error_from_optimization_target(self, df):
        """Get the error of the points so far from the optimization target.
        :param df: a pandas dataframe containing the predictions and the optimization target
        :return df: the same dataframe with an extra column containing the error from the optimization target"""
        df[f'error from optimization target z'] = np.sqrt(np.sum([(alpha * df[f'target {param} z']
                                                                   - alpha * df[f'stzd {param}']) ** 2 for param, alpha
                                                                  in
                                                                  self.alphas.items()], axis=0))
        return df


class ExpectedImprovementConstrained(ExpectedImprovement):
    """This expected improvement class creates an acquisition function where we use the target vector optimisation
     version of the expected improvement for rate and constrain the optimisation with a soft constraint on m, using
     expected feasibility. The final acquisition function is then \alpha = \alpha_r * P(m < threshold_m)."""

    def __init__(self, params,  alphas=None):

        if len(params) < 2:
            raise ValueError('This acquisition function requires two parameters')

        super().__init__(params, alphas)

    def filter_ys(self, ys, targets, params):
        """function to match that of the parent class but just returning ys as they are
        :param ys: candas parameter array of the data
        :param targets: candas parameter array of the targets
        :param params: list of parameters. should be either or both of 'm' and 'r'
        :return: candas parameter array of the data with the drifts (m) below the target drift equal to the target"""

        return ys

    def EI(self, preds, target_parrays, best_yet, params):
        """Expected improvement using target vector optimisation for rate and expected feasibility for drift
        :param preds: a candas uncertain parameter array containing the mean and variance of the predictions
        :param target_parrays: a 2x1 array containing the desired values for the rate and drift parameters
        :param best_yet: the smallest distance between an observation and the target, a float
        :param params: list of parameters. should be either or both of 'm' and 'r'
        :return ei: the expected improvements of each prediction"""

        mu = self.alphas['r'] * np.atleast_2d(preds[f'r_mu_z'].to_numpy()).T
        sig2 = self.alphas['r'] ** 2 * np.atleast_2d(preds[f'r_sig2_z']).T
        target = self.alphas['r'] * np.atleast_2d(target_parrays['r'].z.values())

        mu_m = self.alphas['m'] * np.atleast_2d(preds[f'm_mu_z']).T
        sig2_m = self.alphas['m'] ** 2 * np.atleast_2d(preds[f'm_sig2_z']).T
        threshold_m = self.alphas['m'] * np.atleast_2d(target_parrays['m'].z.values())

        k = 1
        chi_ei = self.Chi_EI(mu, sig2, target, best_yet, k=k)
        ef = self.expected_feasibility(mu_m, sig2_m, threshold_m)
        ei = chi_ei.flatten() * ef.flatten()
        return ei

    def expected_feasibility(self, mu, sig2, threshold_m):
        """Expected feasibility of drift. Calculated as P(m < threshold_m)
        :param mu: array of the means of the drift
        :param sig2: array of the variances of the drift
        :param threshold_m: the threshold value of the drift, standardized"""

        ef = norm.cdf(threshold_m, loc=mu, scale=np.sqrt(sig2))

        return ef

    def BestYet(self, ys, target_parrays):
        """Function to calculate the closest distance from the target we have observed so far, but only for rate as it
        is the only optimisation parameter
        :param ys: a series containing the values in the train set
        :param target_parrays: a 2x1 array containing the desired values for the rate and drift parameters
        :return best: the smallest distance between an observation and the target, a float"""

        rs = self.alphas['r'] * ys['r']
        targ_r = self.alphas['r'] * np.atleast_2d(target_parrays['r'].z.values())

        best = ((rs - targ_r.T) ** 2).min()

        return best

    def get_error_from_optimization_target(self, df):
        """Get the error of the points so far from the optimization target.
        :param df: a pandas dataframe containing the predictions and the optimization target
        :return df: the same dataframe with an extra column containing the error from the optimization target"""

        df['m penalty'] = df['stzd m'] - df['target m z']
        df.loc[df['stzd m'] < df['target m z'], 'm penalty'] = 0
        df[f'error from optimization target z'] = np.sqrt((self.alphas['r'] * df[f'target r z']
                                                           - self.alphas['r'] * df[f'stzd r'])**2) + df['m penalty']
        df = df.drop(columns=['m penalty'])

        return df


