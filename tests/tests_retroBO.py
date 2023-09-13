import unittest
import pytest
from experiments.retro_BO import RetroBO
import numpy as np
import pathlib as pl
import os
import pandas as pd
import warnings
import tensorflow as tf
warnings.simplefilter("ignore")
from candas.learn import ParameterSet, parray, Standardizer
import copy
from pandas.testing import assert_frame_equal
import candas

class DivideDataTests(unittest.TestCase):

    def setUp(self) -> None:
        path = pl.Path(os.getcwd()).parent
        ps_df = pd.read_pickle(path / 'data' / 'ADVI_ParameterSets_220528.pkl')
        ps_df = ps_df[(ps_df.lg10_Copies == 8)]
        ps_df = ps_df.drop(ps_df[ps_df['Experiment'].str.contains("JG073A")].index)
        ps = ParameterSet.from_wide(ps_df)
        ps.data['EvaGreen'] = ((ps.data['Reporter'] == "EVAGREEN") | (ps.data['Reporter'] == "SYBR"))
        ps.data.loc[ps.data['EvaGreen'] == True, 'EvaGreen'] = 'EvaGreen'
        ps.data.loc[ps.data['EvaGreen'] == False, 'EvaGreen'] = 'Probe'
        ps.data['PrimerPairReporter'] = ps.data[['PrimerPair', 'EvaGreen']].agg('-'.join, axis=1)
        self.ps = ps
        self.Xvalid = RetroBO(['BP', 'GC'], ['PrimerPairReporter'], 2, ['r', 'm'], ['lmc'])

    def test_divide_data_warm_start_error(self):
        """test that if n_train < n functions and warm start an error is raised"""
        with self.assertRaises(ValueError):
            train_ps, test_ps = self.Xvalid.divide_data('ADVI_ParameterSets_220528.pkl', seed=None, n_train=3, pct_train=None,
                                            initial_surfaces=None, warm_start=True)

    def test_divide_data_initial_surfaces_error(self):
        """test that if the amount of data in initial surfaces is larger n_train an error is raised"""
        with self.assertRaises(ValueError):
            train_ps, test_ps = self.Xvalid.divide_data('ADVI_ParameterSets_220528.pkl', seed=None, n_train=10, pct_train=None,
                                            initial_surfaces={'FP004-RP004-Probe': 'all', 'FP001-RP001-Probe': 3},
                                            warm_start=True)

    def test_divide_data_initial_surface_error_2(self):
        """test that if no_points specified for a surface in initial surfaces exceeds the amount of data on that surface
        an error is raised"""
        with self.assertRaises(ValueError):
            train_ps, test_ps = self.Xvalid.divide_data('ADVI_ParameterSets_220528.pkl', seed=None, n_train=50, pct_train=None,
                                            initial_surfaces={'FP004-RP004-Probe': 'all', 'FP001-RP001-Probe': 24},
                                            warm_start=True)

    def test_divide_data_n_train(self):
        """test we get the right train, test split when using n_train"""
        n_train = 20
        train_ps, test_ps = self.Xvalid.divide_data('ADVI_ParameterSets_220528.pkl', seed=None, n_train=n_train, pct_train=None,
                                        initial_surfaces=None, warm_start=None)
        train_ps.data['PrimerPairReporterBPGC'] = train_ps.data[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg(
            '-'.join, axis=1)

        # check we have correct number of training points
        assert (len(train_ps.data['PrimerPairReporterBPGC'].unique()) == n_train)
        # check all the data is in train or test
        assert (len(train_ps.data) + len(test_ps.data) == len(self.ps.data))
        # check there is no overlap between train and test
        assert (len(train_ps.data.merge(test_ps.data, how='inner')) == 0)

    def test_divide_data_pct_train(self):
        """test we get the right train, test split when using pct_train"""
        pct_train = 0.20
        train_ps, test_ps = self.Xvalid.divide_data('ADVI_ParameterSets_220528.pkl', seed=None, n_train=None, pct_train=pct_train,
                                        initial_surfaces=None, warm_start=None)
        train_ps.data['PrimerPairReporterBPGC'] = train_ps.data[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg(
            '-'.join, axis=1)
        test_ps.data['PrimerPairReporterBPGC'] = test_ps.data[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg(
            '-'.join, axis=1)
        train_ps_len_unique = len(train_ps.data['PrimerPairReporterBPGC'].unique())
        test_ps_len_unique = len(test_ps.data['PrimerPairReporterBPGC'].unique())

        # check we have correct number of training points
        assert (train_ps_len_unique == round(pct_train * (train_ps_len_unique + test_ps_len_unique)))
        # check all the data is in train or test
        assert (len(train_ps.data) + len(test_ps.data) == len(self.ps.data))
        # check there is no overlap between train and test
        assert (len(train_ps.data.merge(test_ps.data, how='inner')) == 0)

    def test_divide_data_initial_points(self):
        """test the initial surfaces have the right amount of data"""
        initial_surfaces = {'FP004-RP004-Probe': 'all', 'FP001-RP001-Probe': 5}

        train_ps, test_ps = self.Xvalid.divide_data('ADVI_ParameterSets_220528.pkl', seed=None, n_train=70, pct_train=None,
                                        initial_surfaces=initial_surfaces, warm_start=None)

        train_ps.data['PrimerPairReporterBPGC'] = train_ps.data[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg(
            '-'.join, axis=1)
        test_ps.data['PrimerPairReporterBPGC'] = test_ps.data[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg(
            '-'.join, axis=1)

        self.ps.data['PrimerPairReporterBPGC'] = self.ps.data[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg(
            '-'.join, axis=1)
        assert (len(train_ps.data[train_ps.data['PrimerPairReporter'] == 'FP001-RP001-Probe'][
                        'PrimerPairReporterBPGC'].unique()) >= 5)
        assert (len(train_ps.data[train_ps.data['PrimerPairReporter'] == 'FP004-RP004-Probe'][
                        'PrimerPairReporterBPGC'].unique()) == len(
            self.ps.data[self.ps.data['PrimerPairReporter'] == 'FP004-RP004-Probe'][
                'PrimerPairReporterBPGC'].unique()))

    def test_divide_data_warm_start(self):
        """test warm start works"""
        train_ps, test_ps = self.Xvalid.divide_data('ADVI_ParameterSets_220528.pkl', seed=None, n_train=70, pct_train=None,
                                        initial_surfaces=None, warm_start=True)

        assert (len(train_ps.data['PrimerPairReporter'].unique()) == len(self.ps.data['PrimerPairReporter'].unique()))

class TestBasicFunctions(unittest.TestCase):
    def setUp(self):
        n_train = 20
        np.random.seed(1)
        tf.random.set_seed(1)
        self.retroBO = RetroBO(['BP', 'GC'], ['PrimerPairReporter'], 2, ['r', 'm'], ['lmc', 'lvm', 'avg', 'mo_indi'], log_t=False)

        self.train_ps, self.test_ps = self.retroBO.divide_data(file_name='ADVI_ParameterSets_220528.pkl', seed=1, n_train=n_train,
                                        pct_train=None, initial_surfaces=None, warm_start=False)
        self.retroBO.make_data_splits()

    def test_filter_param_test_df(self):

        test_df = self.retroBO.filter_param_df('lmc', 'r')

        test = self.retroBO.data_splits['lmc']['test'].data
        test = test[(test['Parameter'] == 'r') & (test['Metric'] == 'mean')]
        test['r'] = test['Value']
        test = test[test_df.columns]
        assert(test_df.equals(test))

    def test_filter_already_observed(self):

        already_observed = self.retroBO.filter_already_observed('lvm')
        observed_surfaces = self.retroBO.data_splits['lmc']['train'].data['PrimerPairReporter'].unique()
        already_observed_surfaces = already_observed.data['PrimerPairReporter'].unique()
        assert(already_observed_surfaces.sort() == observed_surfaces.sort())

class TestModelFitting(unittest.TestCase):

    def setUp(self):
        n_train = 20
        np.random.seed(1)
        tf.random.set_seed(1)
        self.retroBO = RetroBO(['BP', 'GC'], ['PrimerPairReporter'], 2, ['r', 'm'], ['lmc', 'lvm', 'avg', 'mo_indi'])

        self.train_ps, self.test_ps = self.retroBO.divide_data(file_name='ADVI_ParameterSets_220528.pkl', seed=1, n_train=n_train,
                                        pct_train=None, initial_surfaces=None, warm_start=False)
        self.retroBO.make_data_splits()

    def test_make_data_splits(self):
        data_splits = self.retroBO.make_data_splits()
        assert(data_splits['lvm']['train'].data.equals(self.train_ps.data))
        assert (data_splits['lvm']['test'].data.equals(self.test_ps.data))

    def test_latent_dims(self):
        models = self.retroBO.fit_models(n_restarts=1)

        # test latent dims doesnt change
        assert(len(self.retroBO.latent_dims) == 1)

        # test the length of the data in each model is the same
        lengths = []
        lengths_X = []
        for param in ['r', 'm']:
            for mod_name, mod in models.items():
                lengths.append(len(mod[param].data.data))
                if mod_name == 'lvm':
                    lengths_X.append(len(mod[param].model.X_data.numpy()))
                else:
                    lengths_X.append(len(mod[param].model.data[0]))

        assert(len(set(lengths)) == 1)
        assert (len(set(lengths_X)) == 1)

        # test the mo_indi has W=0 kappa=1
        assert((models['mo_indi']['r'].model.kernel.kernels[1].kappa.numpy() == np.ones(
            len(models['mo_indi']['r'].coregion_levels['PrimerPairReporter']), )).all())
        assert ((models['mo_indi']['m'].model.kernel.kernels[1].W.numpy() == np.zeros(
            models['mo_indi']['m'].model.kernel.kernels[1].W.shape)).all())


class TestExpectedImprovementConstrained(unittest.TestCase):
    """test the expected improvement constrained function"""

    def setUp(self):
        np.random.seed(3)
        tf.random.set_seed(3)

        path = pl.Path(os.getcwd()).parent
        ps_df = pd.read_pickle(path / 'data' / 'ADVI_ParameterSets_220528.pkl')
        ps_df = ps_df[(ps_df.lg10_Copies == 8)]
        ps_df = ps_df.drop(ps_df[ps_df['Experiment'].str.contains("JG073A")].index)
        ps = ParameterSet.from_wide(ps_df)

        ps.stdzr.transforms['r'] = [candas.utils.skip, candas.utils.skip]
        ps.stdzr.transforms['m'] = [candas.utils.skip, candas.utils.skip]
        for param in ['r', 'm']:
            ps.stdzr[param] = {
                'μ': ps.data.loc[(ps.data['Parameter'] == param) & (ps.data['Metric'] == 'mean'), 'Value'].mean(),
                'σ': ps.data.loc[(ps.data['Parameter'] == param) & (ps.data['Metric'] == 'mean'), 'Value'].std()}

        ps.data['EvaGreen'] = ((ps.data['Reporter'] == "EVAGREEN") | (ps.data['Reporter'] == "SYBR"))
        ps.data.loc[ps.data['EvaGreen'] == True, 'EvaGreen'] = 'EvaGreen'
        ps.data.loc[ps.data['EvaGreen'] == False, 'EvaGreen'] = 'Probe'
        ps.data['PrimerPairReporter'] = ps.data[['PrimerPair', 'EvaGreen']].agg('-'.join, axis=1)
        self.stdzr = ps.stdzr


        from experiments.expected_improvements import ExpectedImprovementConstrained

        self.ei = ExpectedImprovementConstrained(params=['r', 'm'])

    def test_BestYet(self):
        """test the calculations of the best yet function"""
        y_rs = np.array([0.9, 2.0, 3.0, -0.98])
        y_ms = np.array([0.01, 0.1, 0.4, 0.1])
        ys = {'r': y_rs, 'm': y_ms}
        target_parray = parray(**{'r': 1, 'm': 1e-2}, stdzr=self.stdzr, stdzd=True)
        ys_parray = parray(**{'r': y_rs, 'm': y_ms}, stdzr=self.stdzr, stdzd=True)
        ys = {param: ys_parray[param].z.values() for param in ['r', 'm']}

        best = self.ei.BestYet(ys, target_parray)
        assert(best - 0.01 < 1e-8)

    def test_ChiEI(self):

        mu = np.array([[1, 1, 0, 0, 2, 2, 1.5, 1.5, -1]]).T
        sig2 = np.array([[1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1]]).T
        target = np.array([1])
        best_yet = np.array([0.5])

        chi_ei = self.ei.Chi_EI(mu, sig2, target, best_yet, k=1)

        assert(chi_ei[1] > chi_ei[0])
        assert (chi_ei[0] > chi_ei[2])
        assert (chi_ei[2] == chi_ei[4])
        assert(chi_ei[4] < chi_ei[6])
        assert(np.argmax(chi_ei) == 1)
        assert(np.argmin(chi_ei) == 8)

        pass

    def test_expected_feasibility(self):
        """test the expected feasibility function"""

        mu = np.array([[1, 1, 0, 0, 2, 2, 1.5, 1.5, -1]]).T
        sig2 = np.array([[1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1]]).T
        threshold_m = np.array([1])

        ef = self.ei.expected_feasibility(mu, sig2, threshold_m)

        from math import erf

        erf_array = np.array([erf(threshold_m - mu[i]) for i in range(len(mu))])

        test = 0.5 * (1.0 + erf_array.flatten() / np.sqrt(2.0)*(np.sqrt(sig2.flatten())))

        assert(1 - ef[3] < 1e-2)
        assert(ef[0] == ef[1])
        assert(ef[0] == 0.5)
        assert(ef[2] > ef[0])
        assert(ef[4] < ef[0])
        assert(np.sum(ef.flatten() - test.flatten()) < 1e-6)

    def test_get_error_from_optimization_target(self):
        """test the get error from optimization target function"""

        df = pd.DataFrame({'stzd m': [0, 0, 2, -1, -2, 0, 2],  'target m z': [1]*7,
                           'stzd r': [2, 1, 1, 1, 1, 3, 3],  'target r z': [2]*7})

        df_res = self.ei.get_error_from_optimization_target(df)

        res_array = df_res[f'error from optimization target z'].to_numpy()

        res_array_true = [0, 1, 2, 1, 1, 1, 2]

        assert(np.sum(res_array - res_array_true) < 1e-8)


# class TestBayesOpt(unittest.TestCase):
#
#     def setUp(self):
#         n_train = 20
#         np.random.seed(3)
#         tf.random.set_seed(3)
#         self.retroBO = RetroBO(['BP', 'GC'], ['PrimerPairReporter'], 2, ['r', 'm'], ['lmc', 'lvm', 'avg',
#                                                                                              'mo_indi'])
#         self.train_ps, self.test_ps = self.retroBO.divide_data(file_name='ADVI_ParameterSets_220528.pkl', seed=1, n_train=n_train,
#                                         pct_train=None, initial_surfaces=None, warm_start=False)
#
#         self.retroBO.make_data_splits()
#
#     # def test_BestYet(self):
#     #     y_rs = pd.Series([1.0, 2.0, 3.0])
#     #     y_ms = pd.Series([0.01, 0.1, 0.4])
#     #     ys = {'r': y_rs, 'm': y_ms}
#     #     target = np.array([[1, 0]])
#     #     best = self.retroBO.BestYet(ys, target)
#     #     assert(best - 0.01 < 0.001)
#     #
#     # def test_BestYet_1_param(self):
#     #     y_rs = pd.Series([1.0, 2.0, 3.0])
#     #     ys = {'r': y_rs}
#     #     target = np.array([[0.3]])
#     #     best = self.retroBO.BestYet(ys, target)
#     #     assert(best - 0.7 < 0.001)
#
#     def test_EI(self):
#         mu = np.array([[-1.0, 2.0, 2.5], [0.01, 0.1, 0.4]]).T
#         sig2 = np.array([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]]).T
#         target = np.array([[1, 0]])
#         best_yet = 1.0
#         ei = self.retroBO.EI(mu, sig2, target, best_yet, k=1)
#         self.assertAlmostEqual(sum(ei - np.array([0.00338, 0.23059, 0.05309])), 0, places=4)
#
#     def test_EI_1_param(self):
#         mu = np.array([[-1.0, 2.0, 2.5]]).T
#         sig2 = np.array([[0.1, 0.2, 0.3]]).T
#         target = np.array([[1]])
#         best_yet = 1.0
#         ei = self.retroBO.EI(mu, sig2, target, best_yet, k=1)
#         self.assertAlmostEqual(sum(ei - np.array([0.0001237007, 0.25683, 0.08011])), 0, places=4)

class TestGetMetrics(unittest.TestCase):

    def setUp(self):
        self.seed = 1
        dims = ['BP', 'GC']
        latent_dims = ['PrimerPairReporter']
        params = ['r', 'm']
        self.params = params
        self.dims = dims
        self.latent_dims = latent_dims
        self.model_names = ['lmc', 'avg', 'lvm', 'mo_indi']
        coregion_rank = 2
        pct_train = 0.95
        np.random.seed(3)
        tf.random.set_seed(3)

        surfaces = ['FP004-RP004-EvaGreen', 'FP001-RP001x-Probe']

        drop_surfaces = ['FP002-RP002-EvaGreen', 'FP001-RP005-Probe',
                         'FP005-RP005-Probe', 'FP002-RP006-Probe',
                         'FP002-RP002x-EvaGreen', 'FP004-RP004x-Probe', 'FP004-RP004x-EvaGreen',
                         'FP002-RP002x-Probe', 'FP006-RP006-Probe', 'FP001-RP001-Probe',
                         'FP003-RP008-EvaGreen', 'FP003-RP008x-EvaGreen',
                         'FP057.1.0-RP003x-EvaGreen', 'FP003-RP003-Probe', 'FP003-RP008-Probe',
                         'FP057.1.0-RP003x-Probe', 'FP005-FP004-EvaGreen', 'FP002-RP004-EvaGreen',
                         'FP005-FP001-EvaGreen', 'RP002x-FP004-EvaGreen', 'FP005-FP001-Probe',
                         'RP001x-FP002-Probe', 'RP002x-FP005-Probe',
                         'RP002x-FP002-EvaGreen', 'RP008x-FP005-Probe', 'FP004-FP005-Probe',
                         'RP008x-FP001-EvaGreen', 'FP001-RP004-EvaGreen', 'FP001-RP001-EvaGreen', 'FP002-RP002-Probe']
        initial_surfaces = {surface: 'all' for surface in surfaces}
        warm_start = True
        one_of_each = True

        self.retroBO = RetroBO(dims, latent_dims, coregion_rank, params, self.model_names, log_t=False)

        path = pl.Path(os.getcwd()).parent

        train_ps, test_ps = self.retroBO.divide_data(file_name='ADVI_ParameterSets_220528.pkl', seed=self.seed,
                                                     n_train=None,
                                                     pct_train=pct_train,
                                                     initial_surfaces=initial_surfaces,
                                                     warm_start=warm_start, one_of_each=one_of_each,
                                                     drop_surfaces=drop_surfaces)

        self.retroBO.make_data_splits()
        self.retroBO.filter_targets()
        self.retroBO.fit_models(n_restarts=2)
        self.retroBO.get_predictions()

    def test_mogp_kernel(self):
        print(np.sum(np.sqrt(np.square(self.retroBO.models['mo_indi']['r'].model.kernel.kernels[1].W))))
        print(np.sum(np.sqrt(np.square(self.retroBO.models['mo_indi']['r'].model.kernel.kernels[1].kappa))))
        assert(np.sum(np.sqrt(np.square(self.retroBO.models['mo_indi']['r'].model.kernel.kernels[1].W)))
               - len(self.retroBO.models['mo_indi']['r'].model.kernel.kernels[1].W.numpy()) <0.001)
        assert(np.sum(np.sqrt(np.square(self.retroBO.models['mo_indi']['r'].model.kernel.kernels[1].kappa))) == 4.0)


    def test_get_metrics(self):

        results_dfs, test_dfs = self.retroBO.get_metrics()
                 # check there are the same number of nans for all models
        assert(len(set([len(results_dfs[param][f'{model_name}_train_RMSE'].isna()) for param in ['r', 'm']
                        for model_name in ['lmc', 'lvm', 'avg','mo_indi']])) == 1)
        assert (len(set(
            [len(results_dfs[param][f'{model_name}_test_RMSE'].isna()) for param in ['r', 'm'] for model_name in
             ['lmc', 'lvm', 'avg', 'mo_indi']])) == 1)
        assert (len(set(
            [len(results_dfs[param][f'{model_name}_train_NLPD'].isna()) for param in ['r', 'm'] for model_name in
             ['lmc', 'lvm', 'avg', 'mo_indi']])) == 1)
        assert (len(set(
            [len(results_dfs[param][f'{model_name}_train_RMSE'].isna()) for param in ['r', 'm'] for model_name in
             ['lmc', 'lvm', 'avg', 'mo_indi']])) == 1)

        for model in self.model_names:
            # check the number of points per surface match that of the train and test sets
            for ps_name, ps in {'test': self.retroBO.test_ps, 'train':
                self.retroBO.train_ps}.items():
                unique_locations = ps.data[['PrimerPairReporter', 'BP', 'GC']].drop_duplicates()
                points_per_surface = unique_locations['PrimerPairReporter'].value_counts()
                for pp in points_per_surface.index:
                    assert(points_per_surface[pp] == results_dfs['r'].loc[pp, f'no {ps_name} points'])

            unique_locations = self.retroBO.test_ps.data[['PrimerPairReporter', 'BP', 'GC']].drop_duplicates()
            unique_locations['PrimerPairReporterBPGC'] = unique_locations[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg(
                '-'.join, axis=1)

            # make sure the same points are found in the test_df as test set
            self.maxDiff = None
            for param, test_df in test_dfs.items():
                df = test_df['test']
                df['PrimerPairReporterBPGC'] = df[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg(
                '-'.join, axis=1)
                test1 = df['PrimerPairReporterBPGC'].unique()
                test2 = unique_locations['PrimerPairReporterBPGC'].to_numpy()
                self.assertCountEqual(df['PrimerPairReporterBPGC'].unique(),
                                      unique_locations['PrimerPairReporterBPGC'].to_numpy())

class test_retroBO_run(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.seed = 1
        dims = ['BP', 'GC']
        latent_dims = ['PrimerPairReporter']
        params = ['r']
        self.params = params
        self.dims = dims
        self.latent_dims = latent_dims
        self.model_names = ['lmc', 'avg', 'lvm', 'mo_indi']
        coregion_rank = 2
        pct_train = 0.95
        plot_figures = False
        surfaces = ['FP004-RP004-EvaGreen', 'FP001-RP001x-Probe' ] #

        drop_surfaces = ['FP002-RP002-EvaGreen', 'FP001-RP005-Probe',
                         'FP005-RP005-Probe', 'FP002-RP006-Probe',
                         'FP002-RP002x-EvaGreen', 'FP004-RP004x-Probe', 'FP004-RP004x-EvaGreen',
                         'FP002-RP002x-Probe', 'FP006-RP006-Probe', 'FP001-RP001-Probe',
                         'FP003-RP008-EvaGreen', 'FP003-RP008x-EvaGreen',
                         'FP057.1.0-RP003x-EvaGreen', 'FP003-RP003-Probe', 'FP003-RP008-Probe',
                         'FP057.1.0-RP003x-Probe', 'FP005-FP004-EvaGreen', 'FP002-RP004-EvaGreen',
                         'FP005-FP001-EvaGreen', 'RP002x-FP004-EvaGreen', 'FP005-FP001-Probe',
                         'RP001x-FP002-Probe', 'RP002x-FP005-Probe',
                         'RP002x-FP002-EvaGreen', 'RP008x-FP005-Probe', 'FP004-FP005-Probe',
                         'RP008x-FP001-EvaGreen', 'FP001-RP004-EvaGreen', 'FP001-RP001-EvaGreen','FP002-RP002-Probe']
        initial_surfaces = {surface: 'all' for surface in surfaces}
        warm_start = True
        one_of_each = True

        self.retroBO = RetroBO(dims, latent_dims, coregion_rank, params, self.model_names, log_t=True)

        path = pl.Path(os.getcwd()).parent

        train_ps, test_ps = self.retroBO.divide_data(file_name='ADVI_ParameterSets_220528.pkl', seed=self.seed, n_train=None,
                                                pct_train=pct_train,
                                                initial_surfaces=initial_surfaces,
                                                warm_start=warm_start, one_of_each=one_of_each,
                                                drop_surfaces=drop_surfaces)

        self.retroBO.make_data_splits()
        # self.test_no_target_repeats(self)

        self.original_data = copy.deepcopy(self.retroBO.data_splits)
        self.original_targets = copy.deepcopy(self.retroBO.targets)

        self.results_df, self.predictions = self.retroBO.run_BO(max_iteration=1, n_restarts=2, save=False,
                                                                seed=self.seed)

    def test_no_target_repeats(self):
        """check that each PrimerPairReporter only occurs once in the targets"""
        assert(self.retroBO.targets.duplicated(subset=['PrimerPairReporter']).any() == False)

    def test_correct_number_of_targets(self):
        """test that the number of targets is the same as or less than the number of surfaces in the test set"""
        assert(len(self.retroBO.targets['PrimerPairReporter'].unique())
               <= len(self.retroBO.data_splits['lvm']['test'].data['PrimerPairReporter'].unique()))

    def test_target_rs_in_dfs(self):
        """test that the target rs are the same for all models in the results df"""
        for model in self.model_names:
            for model2 in self.model_names:
                self.assertListEqual(self.results_df[self.results_df['model'] == model].sort_values(by='PrimerPairReporter')['target r'].tolist(),
                       self.results_df[self.results_df['model'] == model2].sort_values(by='PrimerPairReporter')['target r'].tolist())

    def test_EIs_in_dfs(self):
        """test that the EI in the """
        for model in self.model_names:
            res = self.results_df[(self.results_df['model'] == model) & (self.results_df['iteration'] == 1)]
            preds = self.predictions[0][model]
            for i, res_row in res.iterrows():
                test = res_row['EI_z']
                test2 = preds[preds['PrimerPairReporter'] == res_row['PrimerPairReporter']]['EI_z'].max()
                print(test)
                print(test2)
                print('x')
                assert (res_row['EI_z'] == preds[preds['PrimerPairReporter'] == res_row['PrimerPairReporter']]['EI_z'].max()), \
                    f'next point EI is not equal to EI max of preds df for {model}'

    def test_data_update(self):
        for model in self.model_names:
            for param in self.params:
                train1 = self.original_data[model]['train'].data[(self.original_data[model]['train'].data['Parameter'] == param) &
                (self.original_data[model]['train'].data['Metric'] == 'mean')][['BP', 'GC', 'PrimerPairReporter']].drop_duplicates()
                train2 = self.retroBO.data_splits[model]['train'].data[
                    (self.retroBO.data_splits[model]['train'].data['Parameter'] == param) &
                (self.retroBO.data_splits[model]['train'].data['Metric'] == 'mean')][['BP', 'GC', 'PrimerPairReporter']].drop_duplicates()

                test1 = self.original_data[model]['test'].data[
                    (self.original_data[model]['test'].data['Parameter'] == param) &
                    (self.original_data[model]['test'].data['Metric'] == 'mean')][['BP', 'GC', 'PrimerPairReporter']].drop_duplicates()
                test2 = self.retroBO.data_splits[model]['test'].data[
                    (self.retroBO.data_splits[model]['test'].data['Parameter'] == param) &
                    (self.retroBO.data_splits[model]['test'].data['Metric'] == 'mean')][['BP', 'GC', 'PrimerPairReporter']].drop_duplicates()

                # for df in [train1, train2, test1, test2]:
                #     df = df[['Experiment', 'Well'] + self.dims + self.latent_dims + ['Value']]
                #     df[param] = df['Value']
                #     df = df.drop(columns=['Value'])
                test_ = train1[train1['PrimerPairReporter'].isin(self.retroBO.targets['PrimerPairReporter'].unique())]\
                    .value_counts(['PrimerPairReporter'])
                test_2 = train2[train2['PrimerPairReporter'].isin(self.retroBO.targets['PrimerPairReporter'].unique())]\
                    .value_counts(['PrimerPairReporter'])
                test_3 = test1[test1['PrimerPairReporter'].isin(self.retroBO.targets['PrimerPairReporter'].unique())]\
                    .value_counts(['PrimerPairReporter'])
                test_4 = test2[test2['PrimerPairReporter'].isin(self.retroBO.targets['PrimerPairReporter'].unique())]\
                    .value_counts(['PrimerPairReporter'])

                diff = (train1[train1['PrimerPairReporter'].isin(self.retroBO.targets['PrimerPairReporter'].unique())]
                        .value_counts(['PrimerPairReporter'])
                        - train2[train2['PrimerPairReporter'].isin(self.retroBO.targets['PrimerPairReporter'].unique())]
                        .value_counts(['PrimerPairReporter'])).tolist()
                self.assertListEqual(diff, [-1] * len(self.retroBO.targets))

                diff = (test1[test1['PrimerPairReporter'].isin(self.retroBO.targets['PrimerPairReporter'].unique())]
                        .value_counts(['PrimerPairReporter']) - test2[test2['PrimerPairReporter']
                        .isin(self.retroBO.targets['PrimerPairReporter'].unique())]
                        .value_counts(['PrimerPairReporter'])).tolist()
                self.assertListEqual(diff, [1] * len(self.retroBO.targets))


class test_getting_centre_point(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.seed = 1
        dims = ['BP', 'GC']
        latent_dims = ['PrimerPairReporter']
        params = ['r']
        self.params = params
        self.dims = dims
        self.latent_dims = latent_dims
        self.model_names = ['lmc', 'avg']
        self.log_t = False
        coregion_rank = 2
        initial_surfaces = {'FP004-RP004-EvaGreen': 'all', 'FP002-RP002x-Probe': 'all'}
        self.init_surfaces = list(initial_surfaces.keys())

        drop_surfaces = ['FP003-RP008-EvaGreen', 'FP057.1.0-RP003x-Probe', 'FP002-RP006-Probe',
                         'FP057.1.0-RP003x-EvaGreen', 'FP002-RP002-EvaGreen', 'FP006-RP006-Probe',
                         'FP004-RP004x-Probe', 'FP001-RP005-Probe', 'FP001-RP001-EvaGreen', 'FP003-RP008x-EvaGreen',
                         'FP003-RP008-Probe', 'FP002-RP002-Probe', 'FP001-RP001-Probe', 'FP004-RP004x-EvaGreen',
                         'FP005-RP005-Probe', 'FP003-RP003-Probe']

        retroBO1 = RetroBO(dims, latent_dims, coregion_rank, params, self.model_names, self.log_t)

        self.train_ps, test_ps = retroBO1.divide_data(file_name='ADVI_ParameterSets_220528.pkl', seed=self.seed, n_train=None,
                                                      pct_train=0.98,
                                                      initial_surfaces=initial_surfaces,
                                                      warm_start=True, one_of_each=True,
                                                      drop_surfaces=drop_surfaces,
                                                      start_point='centre', log_transform=False)

        retroBO2 = RetroBO(dims, latent_dims, coregion_rank, params, self.model_names, self.log_t)

        train_ps, test_ps = retroBO2.divide_data(file_name='ADVI_ParameterSets_220528.pkl', seed=self.seed,
                                                 n_train=None,
                                                 pct_train=0.0,
                                                 initial_surfaces=initial_surfaces,
                                                 warm_start=False, one_of_each=False,
                                                 drop_surfaces=drop_surfaces,
                                                 start_point='', log_transform=False)
        retroBO2.make_data_splits()
        retroBO2.filter_targets()
        retroBO2.fit_models(n_restarts=1)
        best_initial_df = retroBO2.get_best_initial()
        retroBO2.get_predictions()
        self.next_points = retroBO2.bayes_opt()

    def test_same_centre_point(self):
        BO2_points = self.next_points['lmc'][['BP', 'GC', 'PrimerPairReporter']].drop_duplicates().sort_values(by='BP').reset_index(drop=True)
        BO1_points = self.train_ps.data[['BP', 'GC', 'PrimerPairReporter']].drop_duplicates()
        BO1_points = BO1_points[~BO1_points['PrimerPairReporter'].isin(self.init_surfaces)].sort_values(by='BP').reset_index(drop=True)
        BO2_points = BO2_points.astype({'BP': 'float64', 'GC': 'float64'})
        BO1_points = BO1_points.astype({'BP': 'float64', 'GC': 'float64'})
        assert_frame_equal(BO1_points, BO2_points)

class test_retroBO_termination(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.seed = 1
        dims = ['BP', 'GC']
        latent_dims = ['PrimerPairReporter']
        params = ['r']
        self.params = params
        self.dims = dims
        self.latent_dims = latent_dims
        self.model_names = ['lmc', 'avg', 'lvm', 'mo_indi']
        coregion_rank = 2
        pct_train = 0.98
        plot_figures = False
        surfaces = ['FP004-RP004-EvaGreen', 'FP001-RP001x-Probe']

        drop_surfaces = ['FP004-RP004-Probe', 'FP001-RP001x-EvaGreen', 'FP002-RP002x-EvaGreen',
                          'FP005-FP001-Probe', 'RP001x-FP002-Probe',
                         'RP002x-FP005-Probe', 'FP005-FP004-EvaGreen',
                         'FP001-RP004-EvaGreen', 'FP002-RP004-EvaGreen', 'FP004-FP005-Probe',
                         'RP008x-FP005-Probe', 'FP005-FP001-EvaGreen', 'RP002x-FP004-EvaGreen',
                         'RP008x-FP001-EvaGreen', 'FP001-RP001-Probe', 'FP002-RP002-Probe',
                         'FP001-RP001-EvaGreen', 'FP002-RP002-EvaGreen', 'FP001-RP005-Probe',
                         'FP005-RP005-Probe', 'FP002-RP006-Probe', 'FP006-RP006-Probe',
                         'FP003-RP008-Probe', 'FP002-RP002x-Probe', 'FP004-RP004x-Probe',
                         'FP004-RP004x-EvaGreen', 'FP003-RP008-EvaGreen', 'FP003-RP008x-EvaGreen',
                         'FP057.1.0-RP003x-EvaGreen', 'FP003-RP003-Probe', 'FP057.1.0-RP003x-Probe']
        initial_surfaces = {surface: 'all' for surface in surfaces}
        warm_start = True
        one_of_each = True

        self.retroBO = RetroBO(dims, latent_dims, coregion_rank, params, self.model_names, log_t=True)

        path = pl.Path(os.getcwd()).parent

        train_ps, test_ps = self.retroBO.divide_data(file_name='ADVI_ParameterSets_220528.pkl', seed=self.seed, n_train=None,
                                                pct_train=pct_train,
                                                initial_surfaces=initial_surfaces,
                                                warm_start=warm_start, one_of_each=one_of_each,
                                                drop_surfaces=drop_surfaces)

        self.retroBO.make_data_splits()
        # self.test_no_target_repeats(self)

        self.original_data = copy.deepcopy(self.retroBO.data_splits)
        self.original_targets = copy.deepcopy(self.retroBO.targets)


    def test_termination(self):

        self.results_df, self.predictions = self.retroBO.run_BO(max_iteration=2, n_restarts=1, save=False,
                                                            seed=1)

















