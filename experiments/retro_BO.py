import copy
import os
import pathlib as pl
import pickle
import time
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from candas.learn import ParameterSet
from gpflow.utilities import read_values
from candas.learn import parray
from experiments.x_validation_functions import CrossValidation
from experiments.expected_improvements import ExpectedImprovement, ExpectedImprovementPenalized


class RetroBO(CrossValidation):
    """class for the running of Bayesian Optimisation on the PCR data. Inherits from the cross validation
    class which performs cross validation for each of the models on the data."""

    def __init__(self, dims, latent_dims, coregion_rank, params, model_names, log_t='Transform', alphas=None, ei='MO',
                 random_if_none=False):
        """
        :param dims: input dimensions (should be ['BP', 'GC'])
        :param latent_dims: the latent dimensions (should be ['PrimerPairReporter'])
        :param coregion_rank: the rank of the coregionalisation model
        :param params: the parameter(s) to be optimised should be 'm', 'r' or both
        :param model_names: names of the models to be compared
        :param log_t: whether the data should be log transformed or not
        :param alphas: the alphas to be used in the expected improvement calculation
        """

        super().__init__(dims, latent_dims, coregion_rank, params, model_names)

        self.targets = self.load_targets()
        self.data_splits = None
        self.pred_dfs = None
        self.log_t = log_t
        if ei == 'MO':
            self.ei = ExpectedImprovement(params=params, alphas=alphas)
        elif ei == 'penalized':
            self.ei = ExpectedImprovementPenalized(params=params, alphas=alphas)
        self.centre_points = {'BP': 282, 'GC': 0.360511}
        self.seed = None
        self.n_restarts = None
        self.lmls = {mod_name: {} for mod_name in self.model_names}
        self.results_df = None
        self.target_m = 1e-2
        self.random_if_none = random_if_none

    def load_targets(self):
        """function to load the target values for each of the surfaces
        :returns targets: a dataframe of the target values"""

        path = pl.Path(os.getcwd()).parent
        with open(path / 'data' / 'JG067 sequence targets.csv', "rb") as file:
            targets = pd.read_csv(file)
        targets['PrimerPair'] = targets[['FPrimer', 'RPrimer']].agg('-'.join, axis=1)
        targets['EvaGreen'] = ((targets['-Strand Label'] == "None") & (targets['+Strand Label'] == "None"))
        targets.loc[targets['EvaGreen'] == True, 'EvaGreen'] = 'EvaGreen'
        targets.loc[targets['EvaGreen'] == False, 'EvaGreen'] = 'Probe'
        targets['PrimerPairReporter'] = targets[['PrimerPair', 'EvaGreen']].agg('-'.join, axis=1)
        targets = targets.drop_duplicates(subset=['PrimerPairReporter'], keep='first')

        return targets

    def make_data_splits(self):
        """function to create the original split of the data into train and test sets for each model. Each model is
        given the same train and test set to begin with
        :returns data_splits: a dictionary of dataframes of the test and train set for each model"""

        self.data_splits = {mod_name: {'train': None, 'test': None} for mod_name in self.model_names}
        for name in self.model_names:
            self.data_splits[name]['train'] = copy.copy(self.train_ps)
            self.data_splits[name]['test'] = copy.copy(self.test_ps)

        return self.data_splits

    def run_BO(self, max_iteration=100, n_restarts=3, save=False, seed=None, test_type_name=None, plot_figures=False):
        """a function which iteratively runs the whole BO process until all the data is observed on all surfaces for
        all models.
        :param max_iteration: the maximum number of iterations to run the BO for
        :param n_restarts: the number of restarts to use in the optimisation of the hyperparameters
        :param save: whether to save the hyperparameters at each iteration and results at the end
        :param seed: the seed to use for the random number generator
        :param test_type_name: the name of the test type to be used in the file name if saving
        :param plot_figures: whether to plot the models at each iteration
        :returns results_df: a dataframe containing the next point for each surface at each iteration for each model
        and the predicted and true values of r and m at those points"""

        dfs = []
        predictions = {}
        self.seed = seed
        self.n_restarts = n_restarts

        i = 0
        while (sum([len(self.data_splits[mod]['test'].data) for mod in self.model_names]) > 0) & (i < max_iteration):
            print('filter targets')

            self.lmls = {mod_name: {} for mod_name in self.model_names}  # dictionary for saving log marginal likelihoods

            self.print_data_splits(i)

            t1 = time.time()
            self.filter_targets()
            if self.targets.duplicated(subset=['PrimerPairReporter']).any():
                raise ValueError('not implemented for multiple targets on the same surface')
            t2 = time.time()

            print(f'filter targets time={t2 - t1}')
            if len(self.targets) == 0:
                break
            print(f'iteration {i}')
            print('fit models')
            self.fit_models(n_restarts=n_restarts)

            if i == 0:
                best_initial_df = self.get_best_initial()

            if save:
                self.save_hyperparameters(test_type_name, i, seed)

            if plot_figures:
                [self.plot_models(self.model_names, param) for param in self.params]
                plt.show()
            print('get predictions')
            self.get_predictions()
            predictions[i] = self.pred_dfs
            print('get next point')
            t1 = time.time()
            next_points = self.bayes_opt()
            t2 = time.time()
            print(f'bayes opt time={t2 - t1}')
            print('length targets', len(self.targets))
            print([f'length next points {mod_name}: {len(next_p)}' for mod_name, next_p in next_points.items()])

            for mod_name, df in next_points.items():
                df['model'] = mod_name
                df['iteration'] = i + 1
                dfs.append(df)

            results_df = pd.concat(dfs)

            preds = []
            for j, pred in predictions.items():
                for model_name in self.model_names:
                    pred[model_name]['model'] = model_name
                    pred[model_name]['iteration'] = j + 1
                    preds.append(pred[model_name])
            predz = pd.concat(preds)

            for param in self.params:
                results_df[f'error {param}'] = np.abs(results_df[param] - results_df[f'{param}_mu'])
                results_df[f'error from target {param} z'] = np.abs(results_df[f'target {param} z']
                                                                    - results_df[f'{param}_mu_z'])
                results_df[f'error from target {param}'] = np.abs(
                    results_df[f'target {param}'] - results_df[f'{param}_mu'])
                results_df[f'error {param} z'] = np.abs(
                    results_df[f'stzd {param}'] - results_df[f'{param}_mu_z'])

            if best_initial_df is not None:
                results_df = results_df.append(best_initial_df)

            print(results_df['iteration'])

            results_df['initial_surface'] = False
            results_df.loc[results_df['PrimerPairReporter'].isin(list(self.initial_surfaces.keys())), 'initial_surface'] = True

            if save:
                self.save_results(predz, results_df, seed, test_type_name)

            if len(self.data_splits[
                       self.model_names[0]]['test'].data[['PrimerPairReporter', 'BP', 'GC']].drop_duplicates()) <= 1:
                print('ran out of test data points')
                break

            t1 = time.time()
            print('update data splits')
            self.update_data_splits(next_points=next_points)
            t2 = time.time()
            print(f'update data splits time={t2 - t1}')
            i = i + 1

        self.results_df = results_df
        return results_df, predictions

    def save_results(self, predz, results_df, seed, test_type_name):
        """save the results and predictions dataframes to a pickle file
        :param predz: the predictions dataframe
        :param results_df: the results dataframe
        :param seed: the seed to use for the random number generator
        :param test_type_name: the name of the test type to be used in the file name if saving"""
        path = pl.Path(
            os.getcwd()).parent.parent / f'Results/RetroBO/restarts_{int(self.n_restarts)}_3008_penalized/results_df_{test_type_name}_{self.random_if_none}_{seed}.pkl'
        with open(path, "wb") as file:
            pickle.dump(results_df, file)
        path = pl.Path(
            os.getcwd()).parent.parent / f'Results/RetroBO/restarts_{int(self.n_restarts)}_3008_penalized/preds_df_{test_type_name}_{self.random_if_none}_{seed}.pkl'
        with open(path, "wb") as file:
            pickle.dump(predz, file)

    def print_data_splits(self, i):
        """print the length of the train and test data for each model for debugging purposes"""
        for mod_name, splits in self.data_splits.items():
            print(f'{mod_name} {i} length train unique:'
                  f' {len(splits["train"].data[["PrimerPairReporter", "BP", "GC"]].drop_duplicates())},'
                  f' length test unique: {len(splits["test"].data[["PrimerPairReporter", "BP", "GC"]].drop_duplicates())}')

    def get_best_initial(self):
        """get the distance from the target for the best initial point for each model. These should all be the same"""

        print('get best initial')
        best_points = []
        pprs = self.targets['PrimerPairReporter'].to_list()
        targs_names = self.targets['Sequence Name'].to_list()

        for name, model in self.models.items():
            for i, ppr in enumerate(pprs):
                train_dfs = []
                for param in self.params:
                    train_df = self.filter_param_df(name, param, train_or_test='train')
                    train_df = train_df[train_df['PrimerPairReporter'] == ppr]
                    if len(train_df) != 0:
                        true_values = train_df[param].to_numpy()
                        true_parray = model[param].parray(**{param: true_values})
                        train_df[f'stzd {param}'] = true_parray[param].z.values()
                        train_df['model'] = name
                        train_df['iteration'] = 0
                        train_df['Sequence Name'] = targs_names[i]

                        # ys = self.get_ys(name, ppr, param, 'train')

                        ys_parray = {}

                        ys = self.get_ys(name, ppr, param, 'train', mean=True)
                        ys_parray[param] = model[param].parray(**{param: ys})

                        target_parrays = {}

                        if param == 'r':
                            target_parrays[param] = model['r'].parray(**{'r': self.targets.iloc[i]['Target Rate']})
                        elif param == 'm':
                            target_parrays[param] = model['m'].parray(**{'m': self.target_m}, stdzd=False)

                        train_df = train_df.groupby(['BP', 'GC', 'PrimerPairReporter', 'Sequence Name', 'model']).mean().reset_index()

                        train_df[f'target {param}'] = target_parrays[param].values()
                        train_df[f'target {param} z'] = target_parrays[param].z.values()

                        train_df[f'error from target {param} z'] = np.abs(train_df[f'target {param} z']
                                                                          - train_df[f'stzd {param}'])
                        train_df[f'error from target {param}'] = np.abs(train_df[f'target {param}']
                                                                          - train_df[f'{param}'])
                        train_dfs.append(train_df)
                    else:
                        pass
                if len(train_dfs) != 0:
                    if len(self.params) == 2:
                        df = pd.merge(*train_dfs, on=['Well', 'Sequence Name', 'iteration', 'model'] + self.dims
                                                     + self.latent_dims)
                    else:
                        df = train_dfs[0]
                    df = self.ei.get_error_from_optimization_target(df)
                    best_point = pd.DataFrame(df.iloc[np.argmin(df[f'error from optimization target z'])]).T
                    best_points.append(best_point)

        if len(best_points) != 0:
            best_df = pd.concat(best_points)
            best_df = best_df.reset_index()
            best_df = best_df.drop(columns=['index', 'Well'])

        else:
            best_df = None
            print('length best_df = 0')

        return best_df

    def filter_targets(self):
        """removes a target from the target dataframe if all the data from that surface has already been observed."""

        for ppr in self.targets['PrimerPairReporter'].unique():
            if ppr not in self.data_splits[self.model_names[0]]['test'].data['PrimerPairReporter'].unique():
                self.targets = self.targets[self.targets['PrimerPairReporter'] != ppr]

        for mod in self.model_names:
            if len(set(self.data_splits[mod]['test'].data['PrimerPairReporter'].unique())
                   - set(self.targets['PrimerPairReporter'].unique())) > 0:
                surfaces = set(self.data_splits[mod]["test"].data["PrimerPairReporter"].unique()) \
                           - set(self.targets["PrimerPairReporter"].unique())
                raise ValueError(
                    f'{surfaces} ' 
                    f'in {mod} test set but not in targets')

    def fit_models(self, n_restarts=3):
        """fit all the models to be used for cross validation and save them into models dictionary. Also print the
        time it takes to fit each model.
        :param train_ps: the parameter set to be used for training the models
        :param n_restarts: the number of restarts to be used for fitting the models"""

        for param in self.params:
            if 'avg' in self.model_names:
                t1 = time.time()
                self.models['avg'][param] = self.fit_avg(param, self.data_splits['avg']['train'], n_restarts=n_restarts)
                t2 = time.time()
                print(f'avg fit time={t2 - t1}')
            if 'lvm' in self.model_names:
                t1 = time.time()
                self.models['lvm'][param] = self.fit_lvm(param, self.data_splits['lvm']['train'], MAP=False, n_restarts=n_restarts)
                t2 = time.time()
                print(f'lvm fit time={t2 - t1}')
            if 'mo_indi' in self.model_names:
                t1 = time.time()
                self.models['mo_indi'][param] = self.fit_mo_indi(param, self.data_splits['mo_indi']['train'], n_restarts=n_restarts)
                t2 = time.time()
                print(f'mo_indi fit time={t2 - t1}')
            if 'lmc' in self.model_names:
                t1 = time.time()
                self.models['lmc'][param] = self.fit_lmc(param, self.data_splits['lmc']['train'], n_restarts=n_restarts)
                t2 = time.time()
                print(f'lmc fit time={t2 - t1}')

        return self.models


    def save_hyperparameters(self, test_type_name, seed):
        """function to save the hyperparameters of each of the models. I save the results outside the repo to avoid
        it getting too big.
        :param test_type_name: name of the test type
        :param iteration: iteration number
        :param seed: seed number"""

        path = pl.Path(os.getcwd()) / f'hyperparameters/restarts_{self.n_restarts}'
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            for param in self.params:
                hyp_df = pd.DataFrame()
                hyp_df['seed'] = [seed]
                hyp_df['parameter array'] = [model[param].data]
                hyp_df['test parray'] = [self.data_splits[name]['test']]
                hyp_df['lmls'] = [self.lmls[name][param]]
                hyp_df['hyperparameters'] = [read_values(model[param].model)]

                if name == 'lvm':
                    hyp_df['train data X'] = [model[param].model.X_data.numpy()]
                    hyp_df['train data X fn'] = [model[param].model.X_data_fn.numpy()]
                    hyp_df['train data y'] = [model[param].model.data.numpy()]
                    hyp_df['H mean'] = [model[param].model.H_data_mean.numpy()]
                    hyp_df['H var'] = [model[param].model.H_data_var.numpy()]

                else:
                    hyp_df['train data X'] = [model[param].model.data[0].numpy()]
                    hyp_df['train data y'] = [model[param].model.data[1].numpy()]

                hyp_df.to_pickle(path /
                                 f'hyperparameters_{name}_{test_type_name}_{self.random_if_none}_{param}__seed_{seed}_{self.n_restarts}.pkl')
        return

    def filter_param_df(self, model_name, param, train_or_test='test'):
        """function to filter out just the correct parameter values from the given model's train or test set
        :param model_name: name of the model
        :param param: name of the parameter
        :param train_or_test: whether to filter the train or test set
        :return test_df: a dataframe of the filtered test set"""
        test_df = self.data_splits[model_name][train_or_test].data[
            (self.data_splits[model_name][train_or_test].data['Parameter'] == param) &
            (self.data_splits[model_name][train_or_test].data['Metric'] == 'mean')]
        test_df = test_df[['Experiment', 'Well'] + self.dims + self.latent_dims + ['Value']]
        test_df[param] = test_df['Value']
        test_df = test_df.drop(columns=['Value'])
        return test_df

    def filter_already_observed(self, mod_name):
        """function to identify which of the surfaces in the test set have previously had a point observed on them
        :return already_observed_ps: a parameter set of the already observed surfaces"""
        already_observed = self.data_splits[mod_name]['test'].data[
            self.data_splits[mod_name]['test'].data['PrimerPairReporter'].isin(
                self.data_splits[mod_name]['train'].data['PrimerPairReporter'])]
        if len(already_observed) > 0:
            already_observed_ps = ParameterSet(already_observed)
            already_observed_ps.stdzr = self.data_splits[mod_name]['test'].stdzr
        else:
            already_observed_ps = None
        return already_observed_ps

    def get_predictions(self):
        """function to get the predictions of each of the models on the test set"""

        pred_dfs = {mod_name: None for mod_name in self.model_names}

        for name, model in self.models.items():
            t1 = time.time()
            mod_dfs = []

            for param in self.params:

                test_df = self.filter_param_df(name, param)

                true_values = test_df[param].to_numpy()
                true_parray = model[param].parray(**{param: true_values})
                test_df[f'stzd {param}'] = true_parray[param].z.values()

                already_observed_ps = self.filter_already_observed(name)

                model_predict_methods = {'lmc': self.predict_lmc_or_indi, 'lvm': self.predict_lvm,
                                         'avg': self.predict_avg,
                                         'mo_indi': self.predict_lmc_or_indi}

                if name in ['lmc', 'mo_indi']:

                    test_df[f'{param}_mu'] = np.nan
                    test_df[f'{param}_sig2'] = np.nan
                    test_df[f'{param}_mu_z'] = np.nan
                    test_df[f'{param}_sig2_z'] = np.nan

                    if already_observed_ps is not None:
                        predictions = model_predict_methods[name](param, already_observed_ps, model[param])
                    else:
                        predictions = None

                    # get the indexes of surfaces with previously observed points
                    idx = test_df[test_df['PrimerPairReporter'].isin(
                        self.data_splits[name]['train'].data['PrimerPairReporter'])].index
                    test_df = self.fill_df_predictions(test_df, idx, model, name, param, predictions, ps_name=None)

                else:
                    predictions = model_predict_methods[name](param, self.data_splits[name]['test'], model[param])
                    idx = test_df.index
                    test_df = self.fill_df_predictions(test_df, idx, model, name, param, predictions, ps_name=None)
                    if test_df.isna().values.any():
                        print(gpflow.utilities.print_summary(model[param]))
                        raise ValueError(f'NaNs in predictions {name}')


                mod_dfs.append(test_df)
                t2 = time.time()
                print(f'{name} prediction time={t2 - t1}')
            if len(mod_dfs) > 1:
                pred_df = pd.merge(*mod_dfs, on=['Experiment', 'Well'] + self.dims + self.latent_dims)
            else:
                pred_df = mod_dfs[0]
            pred_df = pred_df.groupby(self.dims + self.latent_dims).mean()
            pred_df = pred_df.reset_index()
            pred_df = pred_df.drop(columns='Well')
            if pred_df.isna().values.any():
                print(f'NaNs in predictions {name}')
                print(mod_dfs)
                print(pred_df)
                raise ValueError(f'NaNs in predictions {name}')

            pred_dfs[name] = pred_df

        self.pred_dfs = pred_dfs
        return pred_dfs

    def get_ys(self, mod_name, ppr, param, train_or_test, mean=False):
        """get the y values of one of the datasets.
        :param mod_name: name of the model for which we want the dataset
        :param ppr: the PrimerPairReporter combination for the surface
        :param param: the parameter for which we want the ys. should be 'r' or 'm'
        :param train_or_test: whether we want the ys for the train or the test set
        :return y: a list of output values for the given dataset"""

        ys = self.data_splits[mod_name][train_or_test].data[
            (self.data_splits[mod_name][train_or_test].data['Parameter']
             == param) & (self.data_splits[mod_name][train_or_test].data[
                              'PrimerPairReporter'] == ppr)
            & (self.data_splits[mod_name][train_or_test].data[
                   'Metric'] == 'mean')]
        if mean:
            y = ys.groupby(self.dims + self.latent_dims).mean().reset_index()['Value']
        else:
            y = ys['Value']
        return y

    def bayes_opt(self):
        """performs Bayesian optimisation for each model for each surface
        :return next_points: dictionary of dataframes outlining the next point selected for each surface for each model
        and the true and predicted values at those points"""

        next_points = {name: [] for name in self.model_names}
        next_points_dfs = {name: None for name in self.model_names}
        for name, model in self.models.items():
            pprs = self.targets['PrimerPairReporter'].to_list()
            targs_names = self.targets['Sequence Name'].to_list()
            for i, ppr in enumerate(pprs):
                min_BP = self.targets.iloc[i]['Min BP']

                target_parrays = self.create_target_parrays(i, model)

                # calculate the best distance from target so far
                ys = {}
                ys_parray = {}
                for param in self.params:
                    ys[param] = self.get_ys(name, ppr, param, 'train', mean=True)
                    ys_parray[param] = model[param].parray(**{param: ys[param]})

                if len(ys[self.params[0]]) == 0:
                    best_yet = np.ones(len(self.pred_dfs[name][self.pred_dfs[name]['PrimerPairReporter'] == ppr]), ) * 4
                else:
                    best_yet = self.ei.BestYet({param: ys_parray[param].z.values() for param in self.params},
                                               target_parrays)

                preds = self.pred_dfs[name][self.pred_dfs[name]['PrimerPairReporter'] == ppr]

                # calculate the expected improvement
                ei = self.ei.EI(preds, target_parrays, best_yet, self.params)
                ei[np.isnan(ei)] = 0
                self.pred_dfs[name].loc[self.pred_dfs[name]['PrimerPairReporter'] == ppr, f'EI_z'] = ei

                # get the next point
                if (name in ['lmc', 'mo_indi']) & (ppr not in self.data_splits[name]['train'].data['PrimerPairReporter'].unique()):
                    if self.random_if_none:
                        temp_df = self.pred_dfs[name][self.pred_dfs[name]['PrimerPairReporter'] == ppr].sample(n=1)
                        temp_df['Sequence Name'] = targs_names[i]
                        next_point = temp_df
                    else:
                        next_point = self.get_centre_point(self.pred_dfs, name, model[self.params[0]], ppr, targs_names[i])
                else:
                    next_point = self.get_next_point(self.pred_dfs, name, ppr, targs_names[i], min_BP)
                next_point = next_point.iloc[0].to_frame().transpose()

                for param in self.params:
                    next_point[f'target {param}'] = target_parrays[param].values()
                    next_point[f'target {param} z'] = target_parrays[param].z.values()

                assert ((next_point['EI_z'].max() - self.pred_dfs[name][self.pred_dfs[name]['PrimerPairReporter']
                                                                  == ppr]['EI_z'].max()) < 1e-5), f'next point EI is not ' \
                    f'equal to EI max of preds df. EI = {next_point["EI_z"].max()}, max of preds df' \
                    f' = {self.pred_dfs[name][self.pred_dfs[name]["PrimerPairReporter"]== ppr]["EI_z"].max()}'

                next_points[name].append(next_point)

            if len(next_points[name]) > 1:
                next_points_dfs[name] = pd.concat(next_points[name])
            else:
                next_points_dfs[name] = next_points[name][0]

        self.next_points = next_points_dfs

        return self.next_points

    def create_target_parrays(self, i, model):
        """create parrays for each parameter for the target point
        :param i: the index of the target point
        :param model: the model """
        target_parrays = {}
        for param in self.params:
            if param == 'r':
                target_parrays[param] = model['r'].parray(**{'r': self.targets.iloc[i]['Target Rate']})
            elif param == 'm':
                target_parrays[param] = model['m'].parray(**{'m': self.target_m}, stdzd=False)

            else:
                raise AssertionError(f'param {param} isn\'t r or m')
        return target_parrays

    def get_next_point(self, pred_dfs, mod_name, ppr, target_name, min_BP):
        """function to identify the point with the best EI which has BP over the min BP
        :param mod_name: name of the model
        :param ppr: the PrimerPairReporter combination for the surface
        :param pred_dfs: the dataframe of predictions
        :param target_name: the sequence name of the target
        :param min_BP: the min no. base pairs allowed
        :return next_point: a one line dataframe containing the bes point to sample next"""

        temp_df = pred_dfs[mod_name][pred_dfs[mod_name]['PrimerPairReporter'] == ppr].sample(frac=1)
        temp_df['Sequence Name'] = target_name
        arg_max = np.argmax(temp_df[f'EI_z'])
        next_point = pd.DataFrame(temp_df.iloc[arg_max]).T
        return next_point

    def get_centre_point(self, pred_dfs, mod_name, model, ppr, target_name):
        """gets the most central point on the surface of a given primer pair reporter surface
        :param pred_dfs: the dataframe of predictions
        :param mod_name: the name of the model
        :param model: the GP model
        :param ppr: the PrimerPairReporter combination for the surface
        :param target_name: the sequence name of the target
        :return next_point: a one line dataframe containing the most cetral point"""

        temp_df = pred_dfs[mod_name][pred_dfs[mod_name]['PrimerPairReporter'] == ppr].sample(frac=1)

        temp_parray = model.parray(**{'BP': temp_df['BP'], 'GC': temp_df['GC'],
                                'PrimerPairReporter': temp_df['PrimerPairReporter']})

        temp_df['centre dist'] =  np.sqrt((temp_parray['BP'].z.values()) ** 2 + (temp_parray['GC'].z.values()) ** 2)
        temp_df['BP_z'] = temp_parray['BP'].z.values()
        temp_df['GC_z'] = temp_parray['GC'].z.values()
        temp_df['Sequence Name'] = target_name
        temp_df_sorted = temp_df.sort_values(by='centre dist')
        arg_min = np.argmin(temp_df_sorted['centre dist'])
        next_point = pd.DataFrame(temp_df_sorted.iloc[arg_min]).T
        next_point = next_point.drop(columns=['centre dist', 'BP_z', 'GC_z'])
        return next_point

    def update_data_splits(self, next_points):
        """function to update the train and test datasets for each model with the next points. This function removes the
        next points for each model from the test set and adds them to the train set.
        :param next_points: a dictionary of dataframes containing the next points to be sampled for each of the
        models"""

        for name in self.model_names:
            targ_names = self.targets['Sequence Name'].to_list()
            print(f'length targ_names: {len(targ_names)}')
            for i, targ_name in enumerate(targ_names):
                next_point = pd.DataFrame(next_points[name][next_points[name]['Sequence Name'] == targ_name])

                ppr = next_point['PrimerPairReporter'].reset_index()['PrimerPairReporter'][0]
                BP = next_point['BP'][next_point.index].to_numpy()[0]
                GC = next_point['GC'][next_point.index].to_numpy()[0]

                test_rows = self.data_splits[name]['test']
                test_rows = test_rows.data[(test_rows.data['PrimerPairReporter'] == ppr)
                                           & (np.abs(test_rows.data['BP'] - BP) <= 1e-6)
                                           & (np.abs(test_rows.data['GC'] - GC) <= 1e-6)]
                self.data_splits[name]['test'].data = self.data_splits[name]['test'].data.drop(test_rows.index)
                self.data_splits[name]['train'].data = self.data_splits[name]['train'].data.append(test_rows)
                self.data_splits[name]['test'].stdzr = self.stdzr
                self.data_splits[name]['train'].stdzr = self.stdzr


