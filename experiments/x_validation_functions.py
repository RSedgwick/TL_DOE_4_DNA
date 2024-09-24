import candas.learn
import gpflow
import numpy as np
import pathlib as pl
import os
import pandas as pd
import warnings

from gpflow.utilities import read_values
warnings.simplefilter("ignore")
from candas.learn import ParameterSet, parray
import tensorflow as tf
from candas.learn import GP_gpflow, LVMOGP_GP
import copy
from gpflow.config import default_float
import time
import matplotlib.pyplot as plt
import seaborn as sns


def _add_initial_surfaces(initial_surfaces, points_per_surface, train_locations, unique_locations):
    """function for adding the initial surfaces to the training set.
    :param initial_surfaces: dict of the surfaces to be included in the training set and the number of points to be
    added. If 'all' then all points on the surface are added.
    :param points_per_surface: dict of the surfaces and the number of points on each surface
    :param train_locations: the unique locations to be included in the training set
    :param unique_locations: the unique locations in the data
    :return: the training locations
    """
    for name, no_points in initial_surfaces.items():
        if no_points == 'all':
            no_points = points_per_surface[name]
        elif no_points > points_per_surface[name]:
            raise ValueError(f'number of points to be observed on {name} exceeds the number of data points')
        train_locs = unique_locations[unique_locations['PrimerPairReporter'] == name].sample(n=no_points)
        train_locations = pd.concat([train_locations,train_locs])
    return train_locations


class CrossValidation:
    """class for running cross validation on the PCR data"""

    def __init__(self, dims, latent_dims, coregion_rank, params, model_names):
        """:param dims: list of dimensions to be included in the model
        :param latent_dims: list of latent dimensions to be included in the model
        :param coregion_rank: the rank of the coregion kernel
        :param params: list of parameters to be predicted
        :param model_names: list of model names to be used"""

        self.dims = dims
        self.latent_dims = latent_dims
        self.coregion_rank = coregion_rank
        self.params = params
        self.model_names = model_names
        self.centre_points = {'BP': 282, 'GC': 0.360511}  # for when we start from central point
        self.initial_surfaces = None
        self.points_per_surface = None
        self.n_train = None
        self.pct_train = None
        self.train_ps = None
        self.test_ps = None
        self.models = {model_name: {param: None for param in self.params} for model_name in self.model_names}
        self.lmls = {mod_name: {} for mod_name in self.model_names}

    def divide_data(self, file_name, seed=None, n_train=None, pct_train=None, initial_surfaces=None, warm_start=None,
                    one_of_each=False, drop_surfaces=None, start_point=None, log_transform=True):
        """function for dividing the data into train and test sets for cross validation
        :param file_name: the name of the file the data should be taken from
        :param seed: random seed
        :param n_train: the number of unique input locations that should be in the training set. There is more than
        one data point at each location
        :param pct_train: the percentage of unique input locations that should be in the training set. Only one of n_train
        and pct_train should be specified
        :param initial_surfaces: dict of surfaces which are to be in the training set with the number of points from that
        surface to be included. If the dict value is 'all' it means all points from that surface will be included in the
        training set
        :param warm_start: bool. Whether at least one input location for each surface should be in the training set.
        :param one_of_each: if this parameter is True then we get 1 point on each surface, except the surfaces included
         in initial_surfaces.
         :param drop_surfaces: list of surfaces to be dropped from the training set
         :param start_point: the method for choosing the starting point: centre, worst point or no first point
         :param log_transform: bool. Whether the data should be log transformed"""

        self.initial_surfaces = initial_surfaces

        if not (n_train is None) ^ (pct_train is None):
            raise ValueError('Exactly one of "n_train" and "pct_train" must be specified')

        if pct_train:
            if (pct_train > 1) ^ (pct_train < 0):
                raise ValueError('pct_train must be between 0 and 1')

        # set seed
        seed = seed if seed is None else seed
        np.random.seed(seed)
        tf.random.set_seed(seed)

        ps = self._create_parameter_set(file_name, log_transform, drop_surfaces)

        test_ = ps.data[(ps.data['Parameter'] == 'r') & (ps.data['Metric'] == 'mean')]

        # get the unique data locations as there are repeats at each point, and randomise the order
        unique_locations = ps.data[['PrimerPairReporter', 'BP', 'GC']].drop_duplicates()
        unique_locations = unique_locations.sample(frac=1).reset_index(drop=True)

        # get the number of points per surface and create a dataframe to store the training locations
        points_per_surface = unique_locations['PrimerPairReporter'].value_counts()
        self.points_per_surface = points_per_surface
        train_locations = pd.DataFrame(columns=unique_locations.columns)

        # add the number of points specified for each of the initial surfaces
        if initial_surfaces:
            train_locations = _add_initial_surfaces(initial_surfaces, points_per_surface, train_locations,
                                                         unique_locations)

        n_train, pct_train = self._get_n_train_and_pct_train(n_train, pct_train, train_locations, unique_locations)

        if n_train > len(unique_locations):
            raise ValueError('n train is larger than unique data locations')

        # add the starting points to the unseen surfaces, if warm_start is True
        if warm_start:
            n_train, train_locations = self._get_starting_points(n_train, ps, start_point, train_locations,
                                                                 unique_locations)

        if one_of_each is False:
            # select the remaining training points
            remaining_train = unique_locations.drop(train_locations.index).sample(n=n_train)
            train_locations = pd.concat([train_locations, remaining_train])

        # get test locations
        test_ps, train_ps = self._get_train_test_parameter_sets(ps, train_locations, unique_locations)

        self.train_ps = train_ps
        self.test_ps = test_ps

        return train_ps, test_ps

    def _get_train_test_parameter_sets(self, ps, train_locations, unique_locations):
        """function for creating the train and test parameter sets based on the training locations. The test set is
        made up of all unique locations not in the training set.
        :param ps: the parameter set of data
        :param train_locations: the locations to be included in the training set
        :param unique_locations: the unique locations in the data"""

        test_locations = unique_locations.drop(train_locations.index)

        # check that there is no overlap between train and test
        if len(train_locations.merge(test_locations, how='inner')) > 0:
            print('train and test have overlap')
        if (len(train_locations) + len(test_locations)) != len(unique_locations):
            print('length of train +test doesnt equal the length of all the data')

        # create test and train parameter sets based on the locations
        train_locations['PrimerPairReporterBPGC'] = train_locations[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg(
            '-'.join, axis=1)
        test_locations['PrimerPairReporterBPGC'] = test_locations[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg(
            '-'.join, axis=1)
        ps.data['PrimerPairReporterBPGC'] = ps.data[['PrimerPairReporter', 'BP', 'GC']].astype(str).agg('-'.join,
                                                                                                        axis=1)
        train = ps.data[ps.data['PrimerPairReporterBPGC'].isin(train_locations['PrimerPairReporterBPGC'])]
        test = ps.data[ps.data['PrimerPairReporterBPGC'].isin(test_locations['PrimerPairReporterBPGC'])]

        # check that there is no overlap between train and test
        if len(train.merge(test, how='inner')) > 0:
            print('train and test have overlap')


        train = train.drop(columns=['PrimerPairReporterBPGC'])
        test = test.drop(columns=['PrimerPairReporterBPGC'])

        # create the parameter sets and ensure the same standardiser is used for both
        train_ps = ParameterSet(train)
        train_ps.stdzr = self.stdzr

        if len(test_locations) == 0:
            test_ps = None
        else:
            test_ps = ParameterSet(test)
            test_ps.stdzr = self.stdzr


        return test_ps, train_ps

    def _get_starting_points(self, n_train, ps, start_point, train_locations, unique_locations):
        """function for adding the starting points to the training set.
        :param n_train: the number of training points
        :param ps: the parameter set of data
        :param start_point: the way to select the starting location. Must be in 'worst_point', 'centre' or ''
        :param train_locations: the unique locations to be included in the training set
        :param unique_locations: the unique locations in the data
        :return: the remaining number of training points and the training locations
        """
        if start_point == 'worst_point':
            sorted_data = pd.merge(ps.data[(ps.data['Parameter'] == 'r') & (ps.data['Metric'] == 'mean')],
                                   unique_locations,
                                   on=['BP', 'GC', 'PrimerPairReporter'], how='left').sort_values(by='Value')
            n_train, train_locations = self._add_starting_points(n_train, sorted_data, train_locations,
                                                                 unique_locations)
        elif start_point == 'centre':
            temp = copy.deepcopy(unique_locations)
            temp_parray = parray(**{'BP': temp['BP'], 'GC': temp['GC'],
                                    'PrimerPairReporter': temp['PrimerPairReporter']}, stdzr=ps.stdzr)

            temp['centre dist'] = np.sqrt((temp_parray['BP'].z.values()) ** 2 + (temp_parray['GC'].z.values()) ** 2)

            temp['BP_z'] = temp_parray['BP'].z.values()
            temp['GC_z'] = temp_parray['GC'].z.values()
            sorted_data = pd.merge(ps.data[(ps.data['Parameter'] == 'r') & (ps.data['Metric'] == 'mean')],
                                   temp, on=['BP', 'GC', 'PrimerPairReporter'],
                                   how='left').sort_values(by='centre dist')
            n_train, train_locations = self._add_starting_points(n_train, sorted_data, train_locations,
                                                                 unique_locations)

        else:
            for name in unique_locations['PrimerPairReporter'].unique():
                if name in train_locations['PrimerPairReporter'].unique():
                    pass
                else:
                    train_locs = unique_locations[unique_locations['PrimerPairReporter'] == name].sample(n=1)
                    train_locations = pd.concat([train_locations, train_locs], ignore_index=True)
                    n_train -= 1
                    if n_train < 0:
                        raise ValueError(
                            'Adding one observation per surface exceeded specified size of training set')
        return n_train, train_locations

    def _add_starting_points(self, n_train, sorted_data, train_locations, unique_locations):
        """function for adding the starting points to the training set, based on the sorted data
        :param n_train: the number of training points
        :param sorted_data: the sorted data
        :param train_locations: the unique locations to be included in the training set
        :param unique_locations: the unique locations in the data
        :return: the remaining number of training points and the training locations"""
        for name in unique_locations['PrimerPairReporter'].unique():
            if name in train_locations['PrimerPairReporter'].unique():
                pass
            else:
                train_locs = sorted_data[sorted_data['PrimerPairReporter'] == name].iloc[0].to_frame().T
                train_loc = unique_locations.loc[(unique_locations['BP'] == train_locs['BP'].to_numpy()[0]) &
                                                 (unique_locations['GC'] == train_locs['GC'].to_numpy()[0]) &
                                                 (unique_locations['PrimerPairReporter']
                                                  == train_locs['PrimerPairReporter'].to_numpy()[0])]
                train_locations = pd.concat([train_locations,train_loc], ignore_index=True)
                n_train -= 1
                if n_train < 0:
                    raise ValueError(
                        'Adding one observation per surface exceeded specified size of training set')

        return n_train, train_locations

    def _get_n_train_and_pct_train(self, n_train, pct_train, train_locations, unique_locations):
        """function for getting the number of training points and the percentage of training points when only one
        of the two is specified.
        :param n_train: the number of training points
        :param pct_train: the percentage of training points
        :param train_locations: the unique locations to be included in the training set
        :param unique_locations: the unique locations in the data
        :return: the number of training points and the percentage of training points"""

        if n_train is None:
            n_train = round((len(unique_locations) - len(train_locations)) * pct_train)
        if pct_train is None:
            pct_train = n_train / (len(unique_locations) - len(train_locations))

        self.n_train = n_train
        self.pct_train = pct_train
        return n_train, pct_train

    def _create_parameter_set(self, file_name, log_transform, drop_surfaces):
        """load and create the parameter set from the data
        :param file_name: the name of the file containing the data
        :param log_transform: whether to log transform the data
        :param drop_surfaces: the surfaces to be dropped from the data
        :return: the parameter set"""
        # load and format the dataframe
        path = pl.Path(os.getcwd()).parent
        ps_df = pd.read_pickle(path / 'data' / file_name)
        ps_df = ps_df[(ps_df.lg10_Copies == 8)]
        ps_df = ps_df.drop(ps_df[ps_df['Experiment'].str.contains("JG073A")].index)
        ps = ParameterSet.from_wide(ps_df)
        if not log_transform:
            ps.stdzr.transforms['r'] = [candas.utils.skip, candas.utils.skip]
            ps.stdzr.transforms['m'] = [candas.utils.skip, candas.utils.skip]
            for param in ['r', 'm']:
                ps.stdzr[param] = {
                    'μ': ps.data.loc[(ps.data['Parameter'] == param) & (ps.data['Metric'] == 'mean'), 'Value'].mean(),
                    'σ': ps.data.loc[(ps.data['Parameter'] == param) & (ps.data['Metric'] == 'mean'), 'Value'].std()}
        else:
            pass
        ps.data['EvaGreen'] = ((ps.data['Reporter'] == "EVAGREEN") | (ps.data['Reporter'] == "SYBR"))
        ps.data.loc[ps.data['EvaGreen'] == True, 'EvaGreen'] = 'EvaGreen'
        ps.data.loc[ps.data['EvaGreen'] == False, 'EvaGreen'] = 'Probe'
        ps.data['PrimerPairReporter'] = ps.data[['PrimerPair', 'EvaGreen']].agg('-'.join, axis=1)
        self.stdzr = ps.stdzr
        if drop_surfaces is not None:
            ps.data = ps.data[~ps.data['PrimerPairReporter'].isin(drop_surfaces)]

        return ps

    def fit_models(self, train_ps, n_restarts=3):
        """fit all the models to be used for cross validation and save them into models dictionary. Also print the
        time it takes to fit each model.
        :param train_ps: the parameter set to be used for training the models
        :param n_restarts: the number of restarts to be used for fitting the models"""

        for param in self.params:
            if 'avg' in self.model_names:
                t1 = time.time()
                self.models['avg'][param] = self.fit_avg(param, train_ps, n_restarts=n_restarts)
                t2 = time.time()
                print(f'avg fit time={t2 - t1}')
            if 'lvm' in self.model_names:
                t1 = time.time()
                self.models['lvm'][param] = self.fit_lvm(param, train_ps, MAP=False, n_restarts=n_restarts)
                t2 = time.time()
                print(f'lvm fit time={t2 - t1}')
            if 'mo_indi' in self.model_names:
                t1 = time.time()
                self.models['mo_indi'][param] = self.fit_mo_indi(param, train_ps, n_restarts=n_restarts)
                t2 = time.time()
                print(f'mo_indi fit time={t2 - t1}')
            if 'lmc' in self.model_names:
                t1 = time.time()
                self.models['lmc'][param] = self.fit_lmc(param, train_ps, n_restarts=n_restarts)
                t2 = time.time()
                print(f'lmc fit time={t2 - t1}')

        return self.models

    def fit_lmc(self, param, train_ps, n_restarts=3):
        """fit the LMC to the training data
        :param param: the parameter to be fitted
        :param n_restarts: the number of restarts to be used for fitting the models"""

        lmcs = []
        lmls = []

        lengthscale_inits = ['random', 'stats'] # two types of lengthscale initialisations

        for restart in range(n_restarts):

            # definte kappa initialisations
            kappa_inits = [np.ones(len(train_ps.data['PrimerPairReporter'].unique()), ),
                           np.random.uniform(1e-6, 2, len(train_ps.data['PrimerPairReporter'].unique())),
                           np.ones(len(train_ps.data['PrimerPairReporter'].unique()), ) * 1e-6, ]

            for lengthscale_init in lengthscale_inits:
                for kappa_init in kappa_inits:

                    t1 = time.time()
                    failure_point = None
                    try:
                        failure_point = 'initialisation'
                        lmc = GP_gpflow(train_ps)
                        failure_point = 'specify model'
                        _ = lmc.specify_model(spatial_dims=self.dims,
                                              coregion_dims=copy.copy(self.latent_dims),
                                              params=param, coregion_rank=self.coregion_rank)

                        # random W initialisation
                        W_init = np.random.uniform(-1, 1,
                                                   (len(lmc.coregion_levels['PrimerPairReporter']), self.coregion_rank))
                        failure_point = 'build model'
                        lmc.build_model(priors=False, W_init=W_init, kappa_init=kappa_init,
                                        lengthscales_init=lengthscale_init)
                        failure_point = 'train model'
                        lmc.train_model()
                        lml = lmc.model.log_marginal_likelihood().numpy()
                        lmls.append(lml)
                        lmcs.append(lmc)
                        print(f'lmc {lengthscale_init} lml {lml}')

                    except Exception as e:
                        print(e)
                        print(f'failure point: {failure_point}')
                        if hasattr(lmc, 'model'):
                            try:
                                gpflow.utilities.print_summary(lmc.model)
                            except:
                                print('print summary failed')

                t2 = time.time()
                print(f'lmc {lengthscale_init} time {t2 - t1}')

        if len(lmcs) == 0:
            raise ValueError('all lmc inits failed')

        best_index = np.argmax(lmls)
        lmc = lmcs[best_index]
        print(f'LMC LML={lmc.model.log_marginal_likelihood().numpy()}')
        self.lmls['lmc'][param] = lmls
        return lmc


    def fit_lvm(self, param, train_ps, MAP=False, n_restarts=3):
        """fit the LVM to the training data
        :param param: the parameter to be fitted
        :param MAP: whether to use MAP
        :param n_restarts: the number of restarts to be used for fitting the models
        """

        lvms = []
        lmls = []

        lengthscale_inits = ['random', 'stats']
        lvm_inits = ['Gpy', 'mo_PCA'] #, 'random'

        for restart in range(n_restarts):
            for lengthscale_init in lengthscale_inits:
                for lvm_init in lvm_inits:
                    try:
                        lvm = LVMOGP_GP(lvmogp_latent_dims=self.coregion_rank, parameter_set=train_ps)
                        _ = lvm.specify_model(spatial_dims=self.dims,
                                              coregion_dims=copy.copy(self.latent_dims),
                                              params=param, coregion_rank=self.coregion_rank)
                        t1 = time.time()
                        with tf.device('/GPU:0'):
                            lvm.build_model(n_u=100, plot_BGPLVM=False, n_restarts=n_restarts,
                                            lengthscales_init=lengthscale_init,
                                            initialisation=lvm_init, priors=False, MAP=False,
                                            set_inducing_points=False, train_inducing=True)
                            t2 = time.time()
                            lvm.train_model()
                            t3 = time.time()
                        if restart==0:
                            print(f'lvm {lengthscale_init} {lvm_init} build time {t2 - t1} train time {t3 - t2}')
                        lml = lvm.model.elbo().numpy()
                        lvms.append(lvm)
                        lmls.append(lml)

                    except Exception as e:
                        print(e)
                        if hasattr(lvm, 'lvmogp'):
                            try:
                                gpflow.utilities.print_summary(lvm.lvmogp)
                            except:
                                print('print summary failed')
        if len(lvms) == 0:
            raise ValueError('all lvm inits failed')

        best_index = np.nanargmax(lmls)
        lvm = lvms[best_index]
        print(f'LVM ELBO={lvm.model.elbo().numpy()}')
        self.lmls['lvm'][param] = lmls
        return lvm

    def fit_avg(self, param, train_ps, n_restarts=3):
        """fit the LMC to the training data
        :param param: the parameter to be fitted
        :param n_restarts: the number of restarts to be used for fitting the models
        """

        avg_gps = []
        lmls = []

        lengthscale_inits = ['random', 'stats']
        for restart in range(n_restarts):
            for lengthscale_init in lengthscale_inits:
                try:
                    avg_gp = GP_gpflow(train_ps)
                    avg_gp.specify_model(spatial_dims=self.dims, params=param)
                    avg_gp.build_model(priors=False, lengthscales_init=lengthscale_init)
                    avg_gp.train_model()
                    lml = avg_gp.model.log_marginal_likelihood().numpy()
                    lmls.append(lml)
                    avg_gps.append(avg_gp)
                except:
                    pass
        if len(avg_gps) == 0:
            raise ValueError('all avg inits failed')

        best_index = np.argmax(lmls)
        avg_gp = avg_gps[best_index]
        print(f'AVG LML={avg_gp.model.log_marginal_likelihood().numpy()}')
        self.lmls['avg'][param] = lmls
        return avg_gp


    def fit_mo_indi(self, param, train_ps, n_restarts=3):
        """fit a multioutput GP to the training data where the hyperparameters are shaerd across surfaces but no
        information is shared between data on the different surfaces
        :param param: the parameter to be fitted
        :param n_restarts: the number of restarts to be used for fitting the models
        """

        mo_indis = []
        lmls = []

        lengthscale_inits = ['random', 'stats']
        for restart in range(n_restarts):
            for lengthscale_init in lengthscale_inits:

                try:
                    mo_indi = GP_gpflow(train_ps)
                    mo_indi.specify_model(spatial_dims=self.dims,
                                          coregion_dims=copy.copy(self.latent_dims),
                                          params=param, coregion_rank=self.coregion_rank)
                    mo_indi.build_model(priors=False, lengthscales_init=lengthscale_init)

                    # set kappa and W to given values and untrainable
                    mo_indi.model.kernel.kernels[1].kappa.assign(
                        tf.ones(len(mo_indi.coregion_levels['PrimerPairReporter']), ))
                    mo_indi.model.kernel.kernels[1].W.assign(tf.zeros(mo_indi.model.kernel.kernels[1].W.shape))
                    gpflow.set_trainable(mo_indi.model.kernel.kernels[1].kappa, False)
                    gpflow.set_trainable(mo_indi.model.kernel.kernels[1].W, False)
                    mo_indi.train_model()
                    lml = mo_indi.model.log_marginal_likelihood().numpy()
                    lmls.append(lml)
                    mo_indis.append(mo_indi)
                except:
                    pass
        if len(mo_indis) == 0:
            raise ValueError('all mo indi inits failed')

        best_index = np.argmax(lmls)
        mo_indi = mo_indis[best_index]
        print(f'MO_INDI LML={mo_indi.model.log_marginal_likelihood().numpy()}')
        self.lmls['mo_indi'][param] = lmls

        return mo_indi

    def get_predictions(self):
        """function to get the predictions of each of the models on the test set"""

        #create dictionary of dataframes of test points
        test_dfs = {param: None for param in self.params}

        for param in self.params:

            test_df = self.test_ps.data[
                (self.test_ps.data['Parameter'] == param) & (self.test_ps.data['Metric'] == 'mean')]
            test_df = test_df[self.dims + self.latent_dims + ['Value']]

            already_observed_ps = self.get_already_observed_surfaces(self.test_ps)

            # define the predict methods
            model_predict_methods = {'lmc': self.predict_lmc_or_indi, 'lvm': self.predict_lvm,
                                     'avg': self.predict_avg,
                                     'mo_indi': self.predict_lmc_or_indi}

            for name, model in self.models.items():

                if name in ['lmc', 'mo_indi']:
                    predictions = model_predict_methods[name](param, already_observed_ps, model[param])
                    idx = test_df[test_df['PrimerPairReporter'].isin(self.train_ps.data['PrimerPairReporter'])].index

                    test_df.loc[idx, f'{name}_mu'] = predictions.μ
                    test_df.loc[idx, f'{name}_sig2'] = predictions.σ2
                    prior_preds = model[param].uparray(name=param, μ=np.array(0), σ2=np.array(1), stdzd=True)
                    test_df[f'{name}_mu'] = test_df[f'{name}_mu'].fillna(prior_preds.μ)
                    test_df[f'{name}_sig2'] = test_df[f'{name}_sig2'].fillna(prior_preds.σ2)

                else:
                    predictions = model_predict_methods[name](param, self.test_ps, model[param])
                    test_df[f'{name}_mu'] = predictions.μ
                    test_df[f'{name}_sig2'] = predictions.σ2

            test_dfs[param] = test_df
            self.pred_dfs = test_dfs

        return test_dfs

    def get_metrics(self):
        """function to get RMSE and NLPD of train and test sets for all models.
        :return results_dfs: dictionary of results dataframes for each parameter. Each dataframe contains the metrics
        for train and test for each surface for each model plus the number of train and test points on each surface.
        :return test_dfs: dictionary of results dataframes for each parameter. Each dataframe contains the metrics for each
        datapoint. """

        test_dfs = {param: {'test': None, 'train': None} for param in self.params}
        results_dfs = {param: None for param in self.params}

        for param in self.params:

            pps = []
            for ps_name, ps in {'test': self.test_ps, 'train': self.train_ps}.items():
                unique_locations = ps.data[['PrimerPairReporter', 'BP', 'GC']].drop_duplicates()
                points_per_surface = unique_locations['PrimerPairReporter'].value_counts()
                points_per_surface = points_per_surface.to_frame(name=f'no {ps_name} points')
                pps.append(points_per_surface)

            results_df = pd.concat(pps, axis=1)
            results_df = results_df.fillna(0)

            for ps_name, ps in {'test': self.test_ps, 'train': self.train_ps}.items():

                df = ps.data[
                    (ps.data['Parameter'] == param) & (ps.data['Metric'] == 'mean')]
                df = df[self.dims + self.latent_dims + ['Value']]

                already_observed_ps = self.get_already_observed_surfaces(ps)

                model_predict_methods = {'lmc': self.predict_lmc_or_indi, 'lvm': self.predict_lvm,
                                         'avg': self.predict_avg,
                                         'mo_indi': self.predict_lmc_or_indi}

                for name, model in self.models.items():

                    true_values = df['Value'].to_numpy()
                    true_parray = model[param].parray(**{param: true_values})
                    df[f'stzd Value {name}'] = true_parray[param].z.values()

                    # make predictions and put in dataframe
                    if name in ['lmc', 'mo_indi']:
                        predictions = model_predict_methods[name](param, already_observed_ps, model[param])
                        idx = df[
                            df['PrimerPairReporter'].isin(self.train_ps.data['PrimerPairReporter'])].index

                        df = self.fill_df_predictions(df, idx, model, name, param, predictions, ps_name)

                    else:
                        predictions = model_predict_methods[name](param, ps, model[param])
                        idx = df.index
                        df = self.fill_df_predictions(df, idx, model, name, param, predictions, ps_name)

                # calculate the metrics

                df = self.fill_df_metrics(df, ps_name)

                for ppr in results_df.index:
                    temp_df = df[df['PrimerPairReporter'] == ppr]
                    for name in self.model_names:
                        results_df = self.fill_results_df(name, ppr, ps_name, results_df, temp_df)

                for name in self.model_names:

                        results_df = self.fill_results_df(name, 'all', ps_name, results_df, df)

                results_dfs[param] = results_df
                test_dfs[param][ps_name] = df

        return results_dfs, test_dfs

    def fill_results_df(self, name, ppr, ps_name, results_df, temp_df):
        """function to fill results dataframe with RMSE and NLPD from a given dataframe
        :param name: name of model
        :param ppr: primer pair reporter
        :param ps_name: name of ps
        :param results_df: dataframe to fill
        :param temp_df: dataframe to get metrics from
        :return results_df: filled dataframe"""

        results_df.loc[ppr, f'{name}_{ps_name}_RMSE'] = np.sqrt(
            np.mean(temp_df[f'{name}_{ps_name}_squared_error'].dropna()))
        results_df.loc[ppr, f'{name}_{ps_name}_NLPD'] = np.mean(
            temp_df[f'{name}_{ps_name}_nlpd'].dropna())
        results_df.loc[ppr, f'{name}_{ps_name}_RMSE_z'] = np.sqrt(
            np.mean(temp_df[f'{name}_{ps_name}_squared_error_z'].dropna()))
        results_df.loc[ppr, f'{name}_{ps_name}_NLPD_z'] = np.mean(
            temp_df[f'{name}_{ps_name}_nlpd_z'].dropna())

        return results_df

    def fill_df_metrics(self, df, ps_name):
        """given a dataframe, calculate the squared error and nlpd for each model and put in dataframe
        :param df: dataframe to put metrics in
        :param name: name of model
        :param ps_name: name of ps (train or test)"""
        for name in self.model_names:
            df[f'{name}_{ps_name}_squared_error'] = np.square(df[f'{name}_{ps_name}_mu'] - df['Value'])
            df[f'{name}_{ps_name}_nlpd'] = self.get_nlpd(df[f'{name}_{ps_name}_mu'],
                                                         df[f'{name}_{ps_name}_sig2'], df['Value'])
            df[f'{name}_{ps_name}_squared_error_z'] = np.square(df[f'{name}_{ps_name}_mu_z']
                                                                - df[f'stzd Value {name}'])
            df[f'{name}_{ps_name}_nlpd_z'] = self.get_nlpd(df[f'{name}_{ps_name}_mu_z'],
                                                           df[f'{name}_{ps_name}_sig2_z'],
                                                           df[f'stzd Value {name}'])
        return df

    def fill_df_predictions(self, df, idx, model, name, param, predictions, ps_name):
        """given a set of predictions, fill the dataframe
        :param df: dataframe to fill
        :param idx: index of the dataframe to fill
        :param model: model to use
        :param name: name of the model
        :param param: parameter to predict
        :param predictions: candas uparray of predictions
        :param ps_name: test or train ps set
        :return: dataframe with predictions filled"""

        if ps_name is None:
            name_str = param
        else:
            name_str = f'{name}_{ps_name}'

        if predictions is not None:

            df.loc[idx, f'{name_str}_mu'] = predictions.μ
            df.loc[idx, f'{name_str}_sig2'] = predictions.σ2
            df.loc[idx, f'{name_str}_mu_z'] = predictions.z.μ
            df.loc[idx, f'{name_str}_sig2_z'] = predictions.z.σ2

        if (name == 'lmc') or (name == 'mo_indi'):
            prior_preds = model[param].uparray(name=param, μ=np.array(0), σ2=np.array(1), stdzd=True)
            df[f'{name_str}_mu'] = df[f'{name_str}_mu'].fillna(prior_preds.μ)
            df[f'{name_str}_sig2'] = df[f'{name_str}_sig2'].fillna(prior_preds.σ2)
            df[f'{name_str}_mu_z'] = df[f'{name_str}_mu_z'].fillna(prior_preds.z.μ)
            df[f'{name_str}_sig2_z'] = df[f'{name_str}_sig2_z'].fillna(prior_preds.z.σ2)
        return df

    def get_already_observed_surfaces(self, ps):
        """get the surfaces that have already have at least 1 observed data point
        :param ps: parameter set
        :return: parameter set with only the surfaces that have been observed"""
        already_observed = ps.data[
            ps.data['PrimerPairReporter'].isin(self.train_ps.data['PrimerPairReporter'])]
        if len(already_observed) > 0:
            already_observed_ps = ParameterSet(already_observed)
            already_observed_ps.stdzr = ps.stdzr
        else:
            already_observed_ps = None
        return already_observed_ps

    def get_nlpd(self, mu, sig2, y_true):
        """ calculate the negative log predictive density
        :param mu: mean of the predictions
        :param sig2: variance of the predictions
        :param y_true: true values
        :return nlpd: negative log predictive density"""
        nlpd = - (-0.5 * np.log(2 * np.pi) - 0.5 * np.log(sig2)
                  - 0.5 * (np.square(y_true.ravel().reshape(len(y_true.ravel()), ) - mu)) / sig2)

        return nlpd

    def predict_lmc_or_indi(self, param, ps, model):
        """function to get predictions for the lmc and indi models
        :param param: the parameter being used
        :param ps: the parameterset to be used
        :param model: the model to be used
        :return predictions: an uncertain parameter array of the predictions at the input points"""

        lmc_test = GP_gpflow(ps, params=param)
        lmc_test.specify_model(params=param, spatial_dims=self.dims,
                               coregion_dims=copy.copy(self.latent_dims), coregion_rank=self.coregion_rank)
        lmc_test.coregion_coords = model.coregion_coords
        test_X, test_y = lmc_test.get_shaped_data()

        test_X = self.format_test_parray(model, ps, test_X)

        predictions = model.predict_points(test_X.ravel())
        return predictions

    def predict_lvm(self, param, ps, model):
        """function to get predictions for the lvm
        :param param: the parameter being used
        :param ps: the parameterset to be used
        :param model: the model to be used
        :return predictions: an uncertain parameter array of the predictions at the input points"""

        lvm_test = LVMOGP_GP(lvmogp_latent_dims=self.coregion_rank, parameter_set=ps)
        lvm_test.specify_model(spatial_dims=self.dims,
                               coregion_dims=copy.copy(self.latent_dims),
                               params=param, coregion_rank=self.coregion_rank)
        lvm_test.coregion_coords = self.models['lvm'][self.params[0]].coregion_coords
        self.create_unseen_levels(ps.data[(ps.data['Parameter'] == param) &
                                          (ps.data['Metric'] == 'mean')],
                                  reporters=True, primers=True)

        test_X, test_y = lvm_test.get_shaped_data()
        test_X = self.format_test_parray(model, ps, test_X)

        predictions = model.predict_points(test_X.ravel())
        if np.isnan(predictions.μ).any():
            print('LVM predictions are nan')
        return predictions

    def format_test_parray(self, model, ps, test_X):
        """formate the test input parray to match that of the model
        :param model: the model to be used
        :param ps: the parameterset to be used
        :param test_X: the test input parray
        :return test_X: the formatted test input parray"""
        if len(ps.data['PrimerPairReporter'].unique()) <= 1:
            value = model.coregion_coords['PrimerPairReporter'][ps.data['PrimerPairReporter'].unique()[0]]
            dict_params = {'GC': test_X['GC'].values(),
                           'BP': test_X['BP'].values(),
                           'PrimerPairReporter': np.array([[value] * len(test_X)]).T}
            test_X = model.parray(**dict_params, stdzd=False)
        return test_X

    def predict_avg(self, param, ps, model):
        """function to get predictions for the average model
        :param param: the parameter being used
        :param ps: the parameterset to be used
        :param model: the model to be used
        :return predictions: an uncertain parameter array of the predictions at the input points"""

        avg_gp_test = GP_gpflow(ps)
        avg_gp_test.specify_model(spatial_dims=self.dims, params=param)
        avg_gp_test.build_model(priors=False, lengthscales_init='random')
        test_X, test_y = avg_gp_test.get_shaped_data()

        predictions = model.predict_points(test_X.ravel())

        return predictions

    def create_unseen_levels(self, targets, reporters=False, primers=False):
        """get latent variables for the surfaces with no observed data for the LVMOGP. This is done by taking a weighted
        average of the latent variables of the surfaces with the same reporter and primer pair (dependent on the bool
        of those two parameters)
        :param targets: the targets to be used
        :param reporters: whether to use reporters
        :param primers: whether to use primers"""

        if primers:
            reporters = True

        for param in self.params:
            lvm = self.models['lvm'][param]
            len_coreg = len(lvm.coregion_coords['PrimerPairReporter'])
            j = 0
            methods = {}
            same_reporter_dict = {}
            if reporters:
                for reporter in targets['EvaGreen'].unique():
                    same_reporter_dict[reporter] = lvm.data.data.loc[
                        lvm.data.data['EvaGreen'] == reporter, 'PrimerPairReporter'].unique()

            if primers:
                same_P_dict = {}

                for FP in np.unique(targets['FPrimer'].tolist() + targets['RPrimer'].tolist()):
                    same_P_dict[FP] = np.unique(
                        lvm.data.data.loc[lvm.data.data['FPrimer'] == FP, 'PrimerPairReporter'].tolist() + \
                        lvm.data.data.loc[lvm.data.data['RPrimer'] == FP, 'PrimerPairReporter'].tolist())

            for i, targ in enumerate(targets['PrimerPairReporter'].unique()):
                method = 'observed data'
                if targ not in list(lvm.coregion_coords['PrimerPairReporter'].keys()):
                    lvm.coregion_coords['PrimerPairReporter'][targ] = len_coreg + j
                    if reporters:
                        reporter = targets.loc[targets['PrimerPairReporter'] == targ, 'EvaGreen'].iloc[0]
                        if primers:
                            FPrimer = targets.loc[targets['PrimerPairReporter'] == targ, 'FPrimer'].iloc[0]
                            RPrimer = targets.loc[targets['PrimerPairReporter'] == targ, 'RPrimer'].iloc[0]
                            same_Primer = same_P_dict[FPrimer].tolist() + same_P_dict[RPrimer].tolist()
                            # same_Primer = same_RP_dict[RPrimer].tolist() + same_FP_dict[FPrimer].tolist()
                            same_primer_and_reporter = list(set(same_reporter_dict[reporter]).intersection(same_Primer))
                            if len(same_primer_and_reporter) < 2:
                                same_primer_and_reporter = same_reporter_dict[reporter]
                                method = 'reporter'
                            method = "reporter + primers"
                            if len(list(set(same_reporter_dict[reporter]).intersection(
                                    same_P_dict[FPrimer].tolist()))) == 0 or \
                                    len(list(set(same_reporter_dict[reporter]).intersection(
                                        same_P_dict[RPrimer].tolist()))) == 0:
                                same_primer_and_reporter = same_reporter_dict[reporter]
                                method = 'reporter'
                        else:
                            same_primer_and_reporter = same_reporter_dict[reporter]
                            method = 'reporter'

                        same_reporter_index = [lvm.coregion_coords['PrimerPairReporter'][rep]
                                               for rep in same_primer_and_reporter]

                        H_means = copy.copy(lvm.model.H_data_mean).numpy()[same_reporter_index, :]
                        H_vars = copy.copy(lvm.model.H_data_var).numpy()[same_reporter_index, :]
                        H_mean = tf.convert_to_tensor(self.weighted_mean(H_means, H_vars), dtype=default_float())
                        H_var = tf.convert_to_tensor(self.weighted_variance(H_means, H_vars), dtype=default_float())
                    else:
                        H_mean = tf.zeros((1, lvm.model.H_data_mean.shape[1]), dtype=default_float())
                        H_var = tf.ones((1, lvm.model.H_data_mean.shape[1]), dtype=default_float())
                        method = 'prior'

                    lvm.model.H_data_mean = tf.concat([lvm.model.H_data_mean, H_mean], axis=0)
                    lvm.model.H_data_var = tf.concat([lvm.model.H_data_var, H_var], axis=0)
                    j = j + 1
                methods[targ] = method
                self.models['lvm'][param] = lvm

    def plot_models(self, models, param, stdz=False):
        """plot the surfaces for the models"""

        for model in models:
            for param in param:
                mod = self.models[model][param]
                if model == 'avg':
                    xy_pas, z_upas = self.get_grid_prediction_avg(mod)
                    self.plot_surfaces(xy_pas, z_upas, mod, model, 'all', None, param, stdz=stdz, save=True)
                else:
                    for pp_name, pp in mod.coregion_coords['PrimerPairReporter'].items():
                        xy_pas, z_upas = self.get_grid_prediction(mod)
                        self.plot_surfaces(xy_pas, z_upas, mod, model, pp_name, pp, param, stdz=stdz, save=True)

    def weighted_mean(self, means, variances):
        """calculate the weighted mean
        :param means: the means of the values
        :param variances: the variances of the values
        :return: the weighted mean"""
        weighted_avg = np.sum(means / variances, axis=0) / np.sum(1 / variances, axis=0)
        return np.expand_dims(weighted_avg, axis=0)

    def weighted_variance(self, means, variances):
        """calculate the weighted variances
        :param means: the means of the values
        :param variances: the variances of the values
        :return: the weighted variance"""
        weighted_var = np.var(means, axis=0) + np.mean(variances, axis=0)
        return np.expand_dims(weighted_var, axis=0)

    def get_grid_prediction(self, model):
        """get the grid prediction for the model
        :param model: the model to get the prediction for"""

        xy_pas = {}
        z_upas = {}

        for pp_name, pp in model.coregion_coords['PrimerPairReporter'].items():
            model.spatial_grid(resolution={'BP': 100, 'GC': 100}, at=model.parray(PrimerPairReporter=pp))
            model.predict_grid()

            xy_pa, z_upa = model.get_conditional_prediction()
            xy_pas[pp_name] = xy_pa
            z_upas[pp_name] = z_upa

        return xy_pas, z_upas

    def get_grid_prediction_avg(self, model):
        """get the grid prediction for the model
        :param model: the model to get the prediction for"""

        xy_pas = {}
        z_upas = {}

        model.spatial_grid(resolution={'BP': 100, 'GC': 100})
        model.predict_grid()
        xy_pa, z_upa = model.get_conditional_prediction()
        xy_pas['all'] = xy_pa
        z_upas['all'] = z_upa

        return xy_pas, z_upas

    def print_train_test_split(self):
        """print the train test split and the actual pct train for debugging purposes"""

        print(f'length train unique: {len(self.train_ps.data[["PrimerPairReporter", "BP", "GC"]].drop_duplicates())},'
              f' length test unique: {len(self.test_ps.data[["PrimerPairReporter", "BP", "GC"]].drop_duplicates())},')
        actual_pct_train = len(self.train_ps.data[["PrimerPairReporter", "BP", "GC"]].drop_duplicates()) /\
                           (len(self.train_ps.data[["PrimerPairReporter", "BP", "GC"]].drop_duplicates())
                            + len(self.test_ps.data[["PrimerPairReporter", "BP", "GC"]].drop_duplicates()))
        print(f'actual % train:  '
              f'{actual_pct_train}')

    def plot_surfaces(self, xy_pas, z_upas, model, model_name, pp_name, pp, param, stdz = False, save=False,
                      seed=None):
        fig, axs = plt.subplots(ncols=2, figsize=(20, 5))

        if stdz:
            xy_pa = xy_pas[pp_name]
            z_upa = z_upas[pp_name]
        else:
            xy_pa = xy_pas[pp_name]
            z_upa = z_upas[pp_name]
        contour = self.contourf_uparray(*xy_pa.to_list(), z_upa, ax=axs[0])
        cbar = plt.colorbar(contour, ax=axs[0])
        contour2 = self.contourf_uparray(*xy_pa.to_list(), z_upa, ax=axs[1], uncertainty=True)
        cbar2 = plt.colorbar(contour2, ax=axs[1])

        if model_name in ['lmc', 'mo_indi']:
            indices = np.argwhere(model.model.data[0][:, 2].numpy() == pp)
            if stdz:
                Xs = np.squeeze(model.model.data[0][:, :2].z.numpy()[indices])
            else:
                Xs = np.squeeze(model.model.data[0][:, :2].numpy()[indices])
            Xs = Xs.reshape(len(indices), 2)
        elif model_name == 'avg':
            if stdz:
                Xs = np.squeeze(model.model.data[0].z.numpy())
            else:
                Xs = np.squeeze(model.model.data[0].numpy())
            Xs = Xs.reshape(len(Xs), 2)
        else:
            indices = np.argwhere(model.model.X_data_fn.numpy() == pp)
            if stdz:
                Xs = np.squeeze(model.model.X_data.z.numpy()[indices])
            else:
                Xs = np.squeeze(model.model.X_data.numpy()[indices])
            Xs = Xs.reshape(len(indices), 2)

        axs[0].scatter(Xs[:, 0], Xs[:, 1], marker='x', color='k')
        axs[1].scatter(Xs[:, 0], Xs[:, 1], marker='x', color='red')
        # test = axs[0].get_xticklabels()
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_xticklabels([int(float(txt.get_text()) * 100) for txt in axs[0].get_xticklabels()])
            # ax.set_yticklabels([int(float(txt.get_text())) for txt in axs[0].get_yticklabels()])

        axs[0].set_title(model_name + ' mean')
        axs[1].set_title(model_name + ' 2*std')

        # plt.tight_layout()
        plt.suptitle(pp_name)
        if save:
            if model_name == 'avg':
                plt.savefig(f'Debugging_Plots/{model_name}_{param}_Prediction', dpi=500)
            else:
                name = pp_name.replace(".", "")
                plt.savefig(f'Debugging_Plots/{model_name}_{param}_{name}_Prediction', dpi=500)
        plt.show()

    def contourf_uparray(self, x_pa, y_pa, z_upa, ax=None, x_scale='standardized',
                         y_scale='standardized', z_scale='natural', uncertainty=False, vminvmax=None,
                         **kwargs):
        ax = plt.gca() if ax is None else ax

        cmap_crest = sns.color_palette("crest", as_cmap=True)

        cmap_flare = sns.color_palette("flare", as_cmap=True)

        if x_scale == 'standardized':
            x = x_pa.z
        elif x_scale == 'transformed':
            x = x_pa.t
        else:
            x = x_pa

        if y_scale == 'standardized':
            y = y_pa.z
        elif y_scale == 'transformed':
            y = y_pa.t
        else:
            y = y_pa

        if z_scale == 'standardized':
            z = z_upa.z
        elif z_scale == 'transformed':
            z = z_upa.t
        else:
            z = z_upa

        if uncertainty:
            if vminvmax is None:
                levels = 10
            else:
                levels = np.linspace(vminvmax[0], vminvmax[1], 16)
            defaults = dict(levels=levels, cmap=cmap_crest)
            contour = ax.contourf(x, y, 2 * z.σ, **{**defaults, **kwargs})
        else:
            if vminvmax is None:
                levels = 16
            else:
                levels = np.linspace(vminvmax[0], vminvmax[1], 16)
            defaults = dict(levels=levels, cmap='pink')
            contour = ax.contourf(x, y, z.μ, **{**defaults, **kwargs})

        ax.set_ylabel(y_pa.names[0])
        ax.set_xlabel(x_pa.names[0])

        # for (axis, scale, array) in zip(['x', 'y'], [x_scale, y_scale], [x_pa, y_pa]):
        #     if scale == 'standardized':
        #         unstandardize_axis_labels(ax, axis, array)
        return contour
    
    def save_hyperparameters(self, test_type_name, seed, n_restarts):
        """function to save the hyperparameters of each of the models. I save the results outside the repo to avoid
        it getting too big.
        :param test_type_name: name of the test type
        :param iteration: iteration number
        :param seed: seed number"""

        path = pl.Path(os.getcwd()) / f'hyperparameters/restarts_{n_restarts}'
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
                                 f'hyperparameters_{name}_{test_type_name}_{param}_seed_{seed}_{n_restarts}.pkl')
        return


