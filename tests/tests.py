import unittest
import pytest
from experiments.x_validation_functions import CrossValidation
import numpy as np
import pathlib as pl
import os
import pandas as pd
import warnings
import tensorflow as tf
warnings.simplefilter("ignore")
from candas.learn import ParameterSet
from candas.learn import parray, Standardizer
from experiments.expected_improvements import ExpectedImprovement, ExpectedImprovement


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
        self.Xvalid = CrossValidation(['BP', 'GC'], ['PrimerPairReporter'], 2, ['r', 'm'], ['lmc'])

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

class TestModelFitting(unittest.TestCase):

    def setUp(self):
        n_train = 20
        np.random.seed(1)
        tf.random.set_seed(1)
        self.Xvalid = CrossValidation(['BP', 'GC'], ['PrimerPairReporter'], 2, ['r', 'm'], ['lmc', 'lvm', 'avg',
                                                                                             'mo_indi'])
        self.train_ps, self.test_ps = self.Xvalid.divide_data('ADVI_ParameterSets_220528.pkl', seed=2, n_train=n_train,
                                                              pct_train=None, initial_surfaces=None, warm_start=None)

        # test latent dims doesnt change
        # test the data in each model is the same
        # test the mo_indi has W=0 kappa=1

    def test_latent_dims(self):
        models = self.Xvalid.fit_models(n_restarts=2, train_ps=self.Xvalid.train_ps)

        # test latent dims doesnt change
        assert(len(self.Xvalid.latent_dims) == 1)

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

class TestGetMetrics(unittest.TestCase):

    def setUp(self):
        n_train = 20
        np.random.seed(3)
        tf.random.set_seed(3)
        self.Xvalid = CrossValidation(['BP', 'GC'], ['PrimerPairReporter'], 3, ['r', 'm'], ['lmc', 'lvm', 'avg',
                                                                                             'mo_indi'])
        self.train_ps, self.test_ps = self.Xvalid.divide_data('ADVI_ParameterSets_220528.pkl', seed=2, n_train=n_train,
                                                              pct_train=None, initial_surfaces=None, warm_start=None)
        models = self.Xvalid.fit_models(self.train_ps, n_restarts=2)

    def test_get_metrics(self):
        results_dfs, test_dfs = self.Xvalid.get_metrics()


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


        # check the number of points per surface match that of the train and test sets
        for ps_name, ps in {'test': self.test_ps, 'train': self.train_ps}.items():
            unique_locations = ps.data[['PrimerPairReporter', 'BP', 'GC']].drop_duplicates()
            points_per_surface = unique_locations['PrimerPairReporter'].value_counts()
            for pp in points_per_surface.index:
                assert(points_per_surface[pp] == results_dfs['r'].loc[pp, f'no {ps_name} points'])

        # check the lengths of the test_dfs are the same for all models
        assert (len(set([len(test_df) for test_df in test_dfs])) == 1)

        unique_locations = self.test_ps.data[['PrimerPairReporter', 'BP', 'GC']].drop_duplicates()
        unique_locations['PrimerPairReporterBPGC'] = unique_locations[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg(
            '-'.join, axis=1)

        # make sure the same points are found in the test_df as test set
        self.maxDiff = None
        for param, test_dfs in test_dfs.items():
            df = test_dfs['test']
            df['PrimerPairReporterBPGC'] = df[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg(
            '-'.join, axis=1)
            test1 = df['PrimerPairReporterBPGC'].unique()
            test2 = unique_locations['PrimerPairReporterBPGC'].to_numpy()
            self.assertCountEqual(df['PrimerPairReporterBPGC'].unique(),
                                  unique_locations['PrimerPairReporterBPGC'].to_numpy())


class TestExpectedImprovements(unittest.TestCase):

    def setUp(self) -> None:
        self.ei = ExpectedImprovement(params=['r', 'm'], alphas={'r':0.5, 'm':0.5})

    def test_filter_ys(self):
        """test that the filtering of the ys is correct. This filtering is to make any drift less than the target drift
        equal to the target drift"""
        stdzr = Standardizer
        ys = parray(**{'r': np.array([1, 2, 3, 4]) , 'm': np.array([1, 2, 3, 4])}, stdzr=stdzr.default())
        targets = parray(**{'r': np.array([2]) , 'm': np.array([3])}, stdzr=stdzr.default())
        params = ['r', 'm']
        filtered_ys = self.ei.filter_ys(ys, targets, params)
        self.assertListEqual(list(filtered_ys['m'].values()), [3, 3, 3, 4])
        self.assertListEqual(list(filtered_ys['r'].values()), list(ys['r'].values()))
        self.assertIsNot(list(filtered_ys['m'].values()), list(ys['m'].values()))


    def test_Chi_EI(self):
        pass

    def test_BestYet(self):
        stdzr = Standardizer
        ys = parray(**{'r': np.array([1, 2, 3, 4]), 'm': np.array([1, 2, 3, 4])}, stdzr=stdzr.default())
        targets = parray(**{'r': np.array([2]), 'm': np.array([3])}, stdzr=stdzr.default())
        params = ['r', 'm']
        best = self.ei.BestYet({param: ys[param].z.values() for param in params}, targets)
        test = (0.5*ys.z.values() - 0.5*targets.z.values())**2
        test1 = test.sum(axis=0).min()
        self.assertEqual(best, test1)

    def test_get_error_from_optimization_target(self):

        df = pd.DataFrame({f'target r z': [1,2, 3], f'target m z': [3, 2, 1], f'stzd r': [1, 1, 1], f'stzd m': [2, 2, 2]})
        df = self.ei.get_error_from_optimization_target(df)
        [self.assertAlmostEqual(df[f'error from optimization target z'].tolist()[i],[0.5, 0.5, 1.11803][i], places=4) for i in range(len(df))]

    def test_EI(self):
        pass





