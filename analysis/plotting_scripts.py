import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import os
import warnings
import candas

warnings.simplefilter("ignore")
import pandas as pd
from candas.learn import ParameterSet, GP_gpflow, LVMOGP_GP
from candas.plotting.regression_plots import unstandardize_axis_labels
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import seaborn as sns
import matplotlib as mpl
import math
import gpflow

mpl.style.use('mystyle.mplstyle')

full_width = 5.5984252
page_height = 7.85
halfwidth = 2.645669


def fit_lmc_from_hps(param_set, hypparams, param):
    """fit an LMC to the saved hyperparameters and data.
    :param param_set: the parameter set
    :param hypparams: dict of the hyperparameters the hyperparameters
    :param param: the parameter to fit
    :return: the fitted model"""
    lmc = GP_gpflow(param_set)
    _ = lmc.specify_model(spatial_dims=['BP', 'GC'],
                          coregion_dims=['PrimerPairReporter'],
                          params=param, coregion_rank=2)
    lmc.build_model(priors=False, W_init=np.random.uniform(-2, 2,
                                                           (len(lmc.coregion_levels['PrimerPairReporter'])
                                                            , 2)),
                    kappa_init=np.ones(len(param_set.data['PrimerPairReporter'].unique()), ),
                    lengthscales_init='random')
    gpflow.utilities.multiple_assign(lmc.model, hypparams)
    return lmc


def fit_lvmogp_from_hps(param_set, hypparams, param):
    """ fit a LVMOGP to the saved hyperparameters and data.
    :param param_set: the parameter set
    :param hypparams: dict of the hyperparameters the hyperparameters
    :param param: the parameter to fit
    :return: the fitted model"""
    model = LVMOGP_GP(lvmogp_latent_dims=2,
                      parameter_set=param_set)
    _ = model.specify_model(spatial_dims=['BP', 'GC'],
                            coregion_dims=['PrimerPairReporter'],
                            params=param, coregion_rank=2)
    model.build_model(n_u=100, plot_BGPLVM=False, n_restarts=1, lengthscales_init='random',
                      initialisation='random', priors=False, MAP=False,
                      set_inducing_points=False, train_inducing=True)
    gpflow.utilities.multiple_assign(model.lvmogp, hypparams)
    model.gp_dict = {'total': model.lvmogp}
    model.model = model.lvmogp

    return model


def fit_avg_from_hps(param_set, hypparams, param):
    """ fit an average GP to the saved hyperparameters and data.
    :param param_set: the parameter set
    :param hypparams: dict of the hyperparameters the hyperparameters
    :param param: the parameter to fit
    :return: the fitted model"""
    avg_gp = GP_gpflow(param_set)
    avg_gp.specify_model(spatial_dims=['GC', 'BP'], params=param)
    avg_gp.build_model(priors=False, lengthscales_init='random')
    gpflow.utilities.multiple_assign(avg_gp.model, hypparams)

    return avg_gp





def get_test_points(path, param_set):
    """Get the test data point locations from the data set
    :param path: path to the data set
    :param param_set: parameter set of the training data"""

    ps_df = pd.read_pickle(path / 'data' / 'ADVI_ParameterSets_220528.pkl')
    ps_df = ps_df[(ps_df.lg10_Copies == 8)]
    ps_df = ps_df.drop(ps_df[ps_df['Experiment'].str.contains("JG073A")].index)
    ps = ParameterSet.from_wide(ps_df)
    ps.data['EvaGreen'] = ((ps.data['Reporter'] == "EVAGREEN") | (ps.data['Reporter'] == "SYBR"))
    ps.data.loc[ps.data['EvaGreen'] == True, 'EvaGreen'] = 'EvaGreen'
    ps.data.loc[ps.data['EvaGreen'] == False, 'EvaGreen'] = 'Probe'
    ps.data['PrimerPairReporter'] = ps.data[['PrimerPair', 'EvaGreen']].agg('-'.join, axis=1)
    total_parameter_array = ps
    test_points = pd.concat([total_parameter_array.data, param_set.data]).drop_duplicates(keep=False)
    test_locations = test_points[['BP', 'GC', 'PrimerPairReporter']].drop_duplicates()
    return test_locations


def plot_all_surfaces(model, pprs, test_locations, model_name, iteration, seed, param, test_name, input_locations=None,
                      save=False):
    """Plot predictions on all surfaces for a given model
    :param model: model
    :param pprs: list of the different surfaces
    :param test_locations: test locations
    :param model_name: name of the model
    :param iteration: iteration of the retroBO run
    :param seed: seed of the retroBO run
    :param param: parameter which is being modelled
    :param input_locations: input locations
    :param save: whether to save the figure
    """

    xy_pas = {}
    z_upas = {}

    fig = plt.figure(figsize=(full_width, page_height))
    subfigs = fig.subfigures(math.ceil(len(pprs) / 2), 2, wspace=0.05)

    limits = None

    i = 0
    for pp_name in pprs.keys():

        axs = subfigs.flatten()[i].subplots(1, 2)

        xy_pas, z_upas, pp = get_prediction(model, model_name, pp_name, pprs, xy_pas, z_upas)

        if pp_name in test_locations['PrimerPairReporter'].unique():
            test_points = test_locations[test_locations['PrimerPairReporter'] == pp_name][['BP', 'GC']]
            test_points = model.parray(**{name: test_points[name] for name in test_points.columns})
        else:
            test_points = None

        plot_surface(xy_pas, z_upas, model, model_name, pp_name, pp, param, test_points, stdz=False, save=False,
                      axs=axs, input_locations=input_locations)

        subfigs.flatten()[i].suptitle(pp_name)
        subfigs.flatten()[i].subplots_adjust(wspace=0.55)
        i += 1
    fig.suptitle(f'{model_name} iteration {iteration} seed {seed}', y=1.05)

    # fig.subplots_adjust(hspace=0.2, wspace=0.2)
    # plt.tight_layout()
    if save:
        path = pl.Path(os.getcwd()) / 'plots'
        plt.savefig(path / f'{model_name}_{test_name}_iteration_{iteration}_seed_{seed}.pdf', bbox_inches='tight')
    plt.show()

def plot_surface(xy_pas, z_upas, model, model_name, pp_name, pp, param, test_points=None, stdz=False, save=False,
                  seed=None, limits=None, axs=None, input_locations=None):

    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=2,
                                figsize=(full_width, halfwidth))
    xy_pa = xy_pas[pp_name]
    z_upa = z_upas[pp_name]

    if not limits:
        limits = {'mu': None, 'sig': None}

    contour = contourf_uparray(*[xy_pa['BP'], xy_pa['GC']], z_upa, ax=axs[0], vminvmax=limits['mu'])
    contour2 = contourf_uparray(*[xy_pa['BP'], xy_pa['GC']], z_upa, ax=axs[1], vminvmax=limits['sig'],
                                uncertainty=True)

    for i, cont in enumerate([contour, contour2]):
        divider = make_axes_locatable(axs[i])
        cax1 = divider.append_axes("right", size="10%", pad=0.1)
        cbar = plt.colorbar(cont, ax=axs[i], cax=cax1, format='%.2g')

    # cbar = plt.colorbar(contour, ax=axs[0])
    # cbar2 = plt.colorbar(contour2, ax=axs[1])

    if model_name in ['lmc', 'mo_indi']:
        indices = np.argwhere(model.X['PrimerPairReporter'].values() == pp)[:, 0]
        Xs = np.squeeze(np.hstack([model.X['BP'].z.values()[indices],
                                   model.X['GC'].z.values()[indices]]))
        Xs = Xs.reshape(len(indices), 2)
    elif model_name == 'avg':
        locs = input_locations[input_locations['PrimerPairReporter'] == pp_name]
        Xs = model.parray(**{'BP': locs['BP'], 'GC': locs['GC']})
        test = np.atleast_2d(Xs[xy_pa.names[0]].z.values())
        Xs = np.squeeze(np.hstack([np.atleast_2d(Xs['BP'].z.values()).T,
                                   np.atleast_2d(Xs['GC'].z.values()).T]))
        # Xs = np.squeeze(model.model.data[0].numpy())
        Xs = Xs.reshape(len(Xs), 2)
    else:
        indices = np.argwhere(model.model.X_data_fn.numpy() == pp)
        Xs = np.squeeze(model.model.X_data.numpy()[indices])
        Xs = Xs.reshape(len(indices), 2)

    axs[0].scatter(Xs[:, 0], Xs[:, 1], marker='o', color='k', zorder=10)
    axs[1].scatter(Xs[:, 0], Xs[:, 1], marker='o', color='k', zorder=10)

    if test_points is not None:
        axs[0].scatter(test_points['BP'].z.values(), test_points['GC'].z.values(),
                       marker='o', facecolors='none', linewidth=1.5, alpha=0.5, color='k', zorder=10)
        axs[1].scatter(test_points['BP'].z.values(), test_points['GC'].z.values(),
                       marker='o', facecolors='none', linewidth=1.5, alpha=0.5, color='k', zorder=10)

    # test = axs[0].get_xticklabels()

    # major_ticks = np.arange(0, 101, 20)
    # minor_ticks = np.arange(0, 101, 5)
    # if limits:
    #     axs[0].set_xticks(limits['BP'].z.values().min())
    # axs[0].set_xticks(np.arange(10, 400, 10), minor=True)
    # axs[0].set_yticks(np.arange(0.05, 0.95, 30)*100)
    # axs[0].set_yticks(np.arange(0.05, 0.95, 10)*100, minor=True)

    for ax in axs:
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xticklabels([int(float(txt.get_text())) for txt in ax.get_xticklabels()])
        ax.set_yticklabels([int(float(txt.get_text()) * 100) for txt in ax.get_yticklabels()])
    #     ax.set_aspect('equal')

    axs[0].set_title(r'$\mu$')
    axs[1].set_title(r'$2\sigma$')
    axs[0].grid(False)
    axs[1].grid(False)

def get_limits(param):

    if param == 'r':
        limits = {'mean': [0, 0.25], 'variance': [0, 0.1]}
    elif param == 'm':
        limits = {'mean': [0, 0.2], 'variance': [0, 0.1]}

    return limits


def get_prediction(model, model_name, pp_name, pprs, xy_pas, z_upas):
    """Get the prediction for a given model and surface
    :param model: model
    :param model_name: name of the model
    :param pp_name: name of the surface
    :param pprs: list of the different surfaces
    :param xy_pas: parameter array of the input coordinates of the surface
    :param z_upas: parameter array of the model predictions"""

    if model_name in ['lvm', 'lmc', 'mo_indi']:
        pp = pprs[pp_name]
        model.spatial_grid(resolution={'BP': 100, 'GC': 100},
                           at=model.parray(PrimerPairReporter=pp))
    else:
        pp = None
        model.spatial_grid(resolution={'BP': 100, 'GC': 100})
    model.predict_grid()
    xy_pa, z_upa = model.get_conditional_prediction()
    xy_pas[pp_name] = xy_pa
    z_upas[pp_name] = z_upa

    return xy_pas, z_upas, pp


def contourf_uparray(x_pa, y_pa, z_upa, ax=None, x_scale='standardized',
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

    for (axis, scale, array) in zip(['x', 'y'], [x_scale, y_scale], [x_pa, y_pa]):
        if scale == 'standardized':
            unstandardize_axis_labels(ax, axis, array)
    return contour


def load_and_fit_multiple_models(model_names, test_name, iteration, seed, log_transform=False):
    """Load the hyperparameters and fit model for all the models in model_names
    :param model_names: list of model names
    :param test_name: name of the test
    :param iteration: iteration number
    :param seed: seed number
    :param log_transform: whether to log transform the data or not"""

    models = {}
    for model_name in model_names:
        path = pl.Path.home() / f'Even_Newer_Results/No_Transform/Hyperparameters'
        with open(path / (f'hyperparameters_{model_name}_{test_name}_iteration_{iteration}_seed_{seed}_10.pkl'),
                  "rb") as file:
            hyper_params = pickle.load(file)
        hypparams = hyper_params['hyperparameters'][0]
        param_set = hyper_params['parameter array'][0]

        if not log_transform:
            param_set.stdzr.transforms['r'] = [candas.utils.skip, candas.utils.skip]
            param_set.stdzr.transforms['m'] = [candas.utils.skip, candas.utils.skip]
            for param in ['r', 'm']:
                param_set.stdzr[param] = {'μ': param_set.data.loc[
                    (param_set.data['Parameter'] == param) & (param_set.data['Metric'] == 'mean'), 'Value'].mean(),
                                          'σ': param_set.data.loc[(param_set.data['Parameter'] == param) & (
                                                      param_set.data['Metric'] == 'mean'), 'Value'].std()}
        else:
            pass

        if model_name == 'avg':
            model = fit_avg_from_hps(param_set, hypparams, param)
        elif model_name in ['lmc', 'mo_indi']:
            model = fit_lmc_from_hps(param_set, hypparams, param)
        else:
            model = fit_lvmogp_from_hps(param_set, hypparams, param)
        models[model_name] = model
    return models
