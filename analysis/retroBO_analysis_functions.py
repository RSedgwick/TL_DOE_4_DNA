import numpy as np
import pathlib as pl
import os
import pandas as pd
import warnings

warnings.simplefilter("ignore")

from copy import copy, deepcopy
from candas.learn import GP_gpflow, LVMOGP_GP
from candas.learn import ParameterSet

import pickle
import candas
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('mystyle.mplstyle')

def load_retroBO_data(test_name, restarts, initial_surfaces):
    """Load the results of the RetroBO experiments
    :param test_name: name of the test
    :param restarts: number of restarts
    :return: dataframe of the results"""

    results_list = []

    for seed in range(0, 25):
        path_name = pl.Path.home() / f'RetroBO/restarts_{restarts}/results_df_{test_name}_{seed}.pkl'
        # path_name = path_name_ /f'results_df_{test_name}_{seed}.pkl'
        if os.path.exists(path_name):
            with open(path_name, 'rb') as f:
                try:
                    df = pickle.load(f)
                    df['seed'] = seed
                    df = df[~df['PrimerPairReporter'].isin(initial_surfaces)]
                    results_list.append(df)
                except:
                    print(path_name, 'is empty?')
        # else:
        #     print(path_name, 'doesn\'t exist')

    try:
        results_df = pd.concat(results_list)
    except:
        print(f'{test_name} doesn\'t exist')
        results_df = None

    return results_df


def load_cross_validation_data(params, path_name, pct_trains=None):
    results_list = []
    scores = {}
    if pct_trains == None:
        pct_trains = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    for pct_train in pct_trains:
        for param in params:
            for seed in range(0, 300):
                path_ = f'{path_name}_{pct_train}_{seed}_{param}.pkl'
                if os.path.exists(path_):
                    with open(path_, 'rb') as f:
                        try:
                            df = pickle.load(f)
                        except:
                            print(path_, 'is empty?')

                        df['param'] = param
                        results_list.append(df)
    try:
        results_df = pd.concat(results_list)
    except:
        print(f'{path_name} doesn\'t exist')
        results_df = None

    return results_df


def plot_Xvalid_subplot(all_results_df, param, metric, ax, traintest, legend=False):
    linestyles = {'lmc': 'dashdot', 'mo_indi': 'dotted', 'lvm': 'solid', 'avg': 'dashed'}

    df = all_results_df[all_results_df['param'] == param]
    test2 = df.groupby(['pct_train']).median().drop(columns=['no test points', 'no train points'])
    test3 = df.groupby(['pct_train']).quantile(q=0.05).drop(columns=['no test points', 'no train points'])
    test4 = df.groupby(['pct_train']).quantile(q=0.95).drop(columns=['no test points', 'no train points'])

    model_names = ['mo_indi', 'avg', 'lmc', 'lvm']
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {'lmc': cols[0], 'mo_indi': cols[3], 'lvm': cols[1], 'avg': cols[2]}
    for tt in [traintest]:
        for mod_name in model_names:
            test2.plot(y=[f'{mod_name}_{tt}_{metric}'], ax=ax,
                       color=colors[mod_name], legend=False,
                       linestyle=linestyles[mod_name])

        for mod_name in model_names:

            col = f'{mod_name}_{tt}_{metric}'
            ax.fill_between(test2.index, test3[col].to_numpy(), test4[col].to_numpy(), alpha=0.2,
                            color=colors[mod_name])

        # if (metric == 'RMSE') & (tt == 'test') & (param=='m'):
        #     ax.ylim(0, 0.02)
        ax.set_title(f'{param}')
        if legend:
            ax.legend()
        ax.set_xlabel('% data in training set')
        ax.set_ylabel(metric[:4])

def get_number_of_runs_retroBO(results_df):
    n_runs_df = pd.DataFrame(columns=['seed', 'max iteration'])
    n_runs_df['seed'] = range(1, 21)
    for seed in range(1, 21):
        n_runs_df.loc[n_runs_df['seed'] == seed, 'max iteration'] = results_df[results_df["seed"] == seed][
            "iteration"].max()
        # print(f'{seed} {results_df[results_df["seed"] == seed]["iteration"].max()}')
    n_runs_df = n_runs_df.groupby('seed').mean()
    # n_runs_df = n_runs_df.astype(int)
    return n_runs_df.T


def get_number_of_runs_Xvalid(results_df):
    n_runs_df = pd.DataFrame(columns=['n_runs', 'pct_train'])
    n_runs_df['pct_train'] = results_df['pct_train'].unique()
    for pct_train in results_df['pct_train'].unique():
        n_runs_df.loc[n_runs_df['pct_train'] == pct_train, 'n_runs'] = len(
            results_df[results_df['pct_train'] == pct_train][
                "seed"].unique())
        # print(f'{seed} {results_df[results_df["seed"] == seed]["iteration"].max()}')
    n_runs_df = n_runs_df.groupby('pct_train').mean()
    n_runs_df = n_runs_df.astype(int)
    return n_runs_df.T


def get_initial_surfaces_one(test_name, learning_surface_name):
    surfs = ['FP004-RP004-Probe', 'FP001-RP001x-EvaGreen', 'FP002-RP002x-EvaGreen',
             'FP001-RP001x-Probe', 'FP005-FP001-Probe', 'RP001x-FP002-Probe',
             'RP002x-FP005-Probe', 'FP005-FP004-EvaGreen', 'RP002x-FP002-EvaGreen',
             'FP001-RP004-EvaGreen', 'FP002-RP004-EvaGreen', 'FP004-FP005-Probe',
             'RP008x-FP005-Probe', 'FP005-FP001-EvaGreen', 'RP002x-FP004-EvaGreen',
             'RP008x-FP001-EvaGreen']
    surfs.remove(learning_surface_name)
    return surfs


def get_initial_surfaces(test_name, learning_surface_name):
    if "one_from_many" in test_name:
        if learning_surface_name == '_FP002-RP002x-EvaGreen':
            initial_surfaces = ['FP001-RP001-Probe', 'FP002-RP002-Probe', 'FP004-RP004-EvaGreen',
                                'FP001-RP001-EvaGreen', 'FP002-RP002-EvaGreen', 'FP001-RP005-Probe',
                                'FP005-RP005-Probe', 'FP002-RP006-Probe', 'FP006-RP006-Probe',
                                'FP003-RP008-Probe', 'FP002-RP002x-Probe', 'FP004-RP004x-Probe',
                                'FP004-RP004x-EvaGreen', 'FP003-RP008-EvaGreen', 'FP003-RP008x-EvaGreen',
                                'FP057.1.0-RP003x-EvaGreen', 'FP003-RP003-Probe', 'FP057.1.0-RP003x-Probe',
                                'FP001-RP001x-EvaGreen', 'FP004-RP004-Probe',
                                'FP001-RP001x-Probe', 'FP005-FP001-Probe', 'RP001x-FP002-Probe',
                                'RP002x-FP005-Probe', 'FP005-FP004-EvaGreen', 'RP002x-FP002-EvaGreen',
                                'FP001-RP004-EvaGreen', 'FP002-RP004-EvaGreen', 'FP004-FP005-Probe',
                                'RP008x-FP005-Probe', 'FP005-FP001-EvaGreen', 'RP002x-FP004-EvaGreen',
                                'RP008x-FP001-EvaGreen']
        else:
            initial_surfaces = ['FP001-RP001-Probe', 'FP002-RP002-Probe', 'FP004-RP004-EvaGreen',
                                'FP001-RP001-EvaGreen', 'FP002-RP002-EvaGreen', 'FP001-RP005-Probe',
                                'FP005-RP005-Probe', 'FP002-RP006-Probe', 'FP006-RP006-Probe',
                                'FP003-RP008-Probe', 'FP002-RP002x-Probe', 'FP004-RP004x-Probe',
                                'FP004-RP004x-EvaGreen', 'FP003-RP008-EvaGreen', 'FP003-RP008x-EvaGreen',
                                'FP057.1.0-RP003x-EvaGreen', 'FP003-RP003-Probe', 'FP057.1.0-RP003x-Probe',
                                'FP001-RP001x-EvaGreen', 'FP002-RP002x-EvaGreen',
                                'FP001-RP001x-Probe', 'FP005-FP001-Probe', 'RP001x-FP002-Probe',
                                'RP002x-FP005-Probe', 'FP005-FP004-EvaGreen', 'RP002x-FP002-EvaGreen',
                                'FP001-RP004-EvaGreen', 'FP002-RP004-EvaGreen', 'FP004-FP005-Probe',
                                'RP008x-FP005-Probe', 'FP005-FP001-EvaGreen', 'RP002x-FP004-EvaGreen',
                                'RP008x-FP001-EvaGreen']
    else:
        initial_surfaces = ['FP004-RP004-EvaGreen', 'FP002-RP002x-Probe']

    return initial_surfaces


def get_colors():
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {'lmc': cols[0], 'mo_indi': cols[3], 'lvm': cols[1], 'avg': cols[2]}
    return colors

def load_parameterset(log_t=False):
    """load the dataset"""

    path = pl.Path(os.getcwd())
    ps_df = pd.read_pickle(path / 'data' / 'ADVI_ParameterSets_220528.pkl')
    ps_df = ps_df[(ps_df.lg10_Copies == 8)]
    ps_df = ps_df.drop(ps_df[ps_df['Experiment'].str.contains("JG073A")].index)
    ps = ParameterSet.from_wide(ps_df)
    ps.data['EvaGreen'] = ((ps.data['Reporter'] == "EVAGREEN") | (ps.data['Reporter'] == "SYBR"))
    ps.data.loc[ps.data['EvaGreen'] == True, 'EvaGreen'] = 'EvaGreen'
    ps.data.loc[ps.data['EvaGreen'] == False, 'EvaGreen'] = 'Probe'
    ps.data['PrimerPairReporter'] = ps.data[['PrimerPair', 'EvaGreen']].agg('-'.join, axis=1)
    if not log_t:
        ps.stdzr.transforms['r'] = [candas.utils.skip, candas.utils.skip]
        ps.stdzr.transforms['m'] = [candas.utils.skip, candas.utils.skip]
        for param in ['r', 'm']:
            ps.stdzr[param] = {
                'μ': ps.data.loc[(ps.data['Parameter'] == param) & (ps.data['Metric'] == 'mean'), 'Value'].mean(),
                'σ': ps.data.loc[(ps.data['Parameter'] == param) & (ps.data['Metric'] == 'mean'), 'Value'].std()}
    return ps

def load_targets():
    """loads the target sets"""

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


def get_best_points(params, results_df, stzd=True, log_t=False):
    """function to get the min difference between data and target for rate and drift
    :param params: list of parameters to be considered
    :param results_df: dataframe with the results
    :param combined: boolean to indicate if combined metric should be used
    :param stzd: boolean to indicate if standardized results should be used
    :param log_t: boolean to indicate if results are log transformed"""

    targets = load_targets()

    cols = ['PrimerPairReporter']

    # if (combined == False) & (stzd == False):
    #     print('must use standardized when calculating combined metric. Changing stzd to True')
    #     stzd = True

    if 'r' in params:
        cols.append('Target Rate')

    best_points = targets[cols]
    best_points = best_points.reset_index()


    if 'm' in params:
        if stzd:
            targ_m = results_df['target m z'].to_numpy()[0]
        else:
            targ_m = 1e-2
        best_points['Target Drift'] = targ_m

    for ppr in targets['PrimerPairReporter'].unique():
        temp = copy(results_df[(results_df['PrimerPairReporter'] == ppr)])
        temp = temp.reset_index()

        if 'r' in params and 'm' in params:
            temp.loc[temp['stzd m'] < temp['target m z'], 'stzd m'] = temp['target m z']
            temp['diff'] = ((temp['stzd r'].to_numpy() - temp['target r z'].to_numpy()) ** 2
                            + (temp['stzd m'].to_numpy() - temp['target m z'].to_numpy()) ** 2) ** (0.5)
            best_points.loc[best_points['PrimerPairReporter'] == ppr, f'min error from targ comb'] = np.min(
                temp['diff'])

        for param in params:

            if param == 'm':
                if stzd:
                    temp.loc[temp['stzd m'] < temp['target m z'], 'stzd m'] = temp['target m z']
                else:
                    temp.loc[temp['m'] < temp['target m'], 'm'] = temp['target m']

            if stzd:
                temp['diff'] = ((temp[f'stzd {param}'].to_numpy() - temp[f'target {param} z'].to_numpy()) ** 2) ** 0.5
            else:
                temp['diff'] = ((temp[param].to_numpy() - temp[f'target {param}'].to_numpy()) ** 2) **0.5

            best_points.loc[best_points['PrimerPairReporter'] == ppr, f'min error from targ {param}'] = np.min(
                temp['diff'])

    return best_points

def get_best_point_penalized(params, results_df, stzd=True, log_t=False):
    """function to get the min difference between data and target for rate and drift
    :param params: list of parameters to be considered
    :param results_df: dataframe with the results
    :param combined: boolean to indicate if combined metric should be used
    :param stzd: boolean to indicate if standardized results should be used
    :param log_t: boolean to indicate if results are log transformed"""

    targets = load_targets()


    cols = ['PrimerPairReporter']

    if 'r' in params:
        cols.append('Target Rate')

    best_points = targets[cols]
    best_points = best_points.reset_index()

    if 'm' in params:
        if stzd:
            targ_m = results_df['target m z'].to_numpy()[0]
        else:
            targ_m = 1e-2
        best_points['Target Drift'] = targ_m

    for ppr in targets['PrimerPairReporter'].unique():
        temp = copy(results_df[(results_df['PrimerPairReporter'] == ppr)])
        temp = temp.reset_index()

        if 'r' in params and 'm' in params:
            temp['m penalty'] = temp['stzd m'] - temp['target m z']
            temp.loc[temp['stzd m'] < temp['target m z'], 'm penalty'] = 0
            temp[f'error from optimization target z'] = np.sqrt((temp[f'target r z'].to_numpy().astype(float)
                                                               - temp[f'stzd r'].to_numpy().astype(float)) ** 2) + temp[
                                                          'm penalty'].to_numpy().astype(float)
            temp[f'error from optimization target r z'] = np.sqrt((temp[f'target r z'].to_numpy().astype(float)
                                                                 - temp[f'stzd r'].to_numpy().astype(float)) ** 2)
            temp[f'error from optimization target m z'] = temp['m penalty'].to_numpy().astype(float)
            temp = temp.drop(columns=['m penalty'])
            best_points.loc[best_points['PrimerPairReporter'] == ppr, f'min error from targ comb'] = np.min(
                temp['error from optimization target z'])
            best_points.loc[best_points['PrimerPairReporter'] == ppr, f'error from targ comb r'] = np.min(
                temp[f'error from optimization target r z'])
            best_points.loc[best_points['PrimerPairReporter'] == ppr, f'error from targ comb m'] = np.min(
                temp[f'error from optimization target m z'])

    return best_points

def get_win_counts(regret_df, regret_metric='combined regret'):
    """get the number of times each model wins (gets to 0 regret first) for both each surface and a total of all"""

    df = regret_df
    win_df = pd.DataFrame(columns=['PrimerPairReporter'] + list(df['model'].unique()))
    win_df['PrimerPairReporter'] = df['PrimerPairReporter'].unique()
    for ppr in df['PrimerPairReporter'].unique():
        winners = []
        for seed in range(1, 21):
            temp_df = df[(df['PrimerPairReporter'] == ppr) & (df['seed'] == seed)]
            if len(temp_df) < 1:
                winner = None
            else:
                zero_regret_dict = {model: None for model in temp_df['model'].unique()}

                for model in temp_df['model'].unique():
                    zero_regret_dict[model] = np.min(
                        temp_df.loc[(temp_df['model'] == model) & (temp_df[regret_metric] <= 1e-6), 'iteration'])
                minval = min(zero_regret_dict.values())
                winner = [k for k, v in zero_regret_dict.items() if v == minval]
                winners.append(winner[:])
            flat_winners_list = [item for sublist in winners for item in sublist]
            for model in temp_df['model'].unique():
                win_df.loc[win_df['PrimerPairReporter'] == ppr, model] = flat_winners_list.count(model)

    win_df_totals = pd.DataFrame(win_df.sum(axis=0)).T[[mod for mod in df['model'].unique()]]
    return win_df, win_df_totals

def calculate_regret(results_df, params, best_points, stzd=True, diff_from_target_only=False, penalized=False):
    """calculate the regret for each model
    :param results_df: the results dataframe
    :param params: the parameters that were optimized
    :param initial_surfaces: the initial surfaces
    :param best_points: the best points dataframe
    :param stzd: whether the results are from stzd or not
    :param combined: whether to calculate the regret for the parameters combined or not"""

    results_df = results_df.reset_index()

    regret_names = []
    if 'm' in params:
        if stzd:
            targ_m = results_df['target m z'].unique()[0]
            results_df.loc[results_df['stzd m'] < results_df['target m z'], 'stzd m'] = results_df['target m z']
        else:
            targ_m = 1e-2
            results_df.loc[results_df['m'] < targ_m, 'm'] = targ_m

    # df = results_df[~results_df['PrimerPairReporter'].isin(initial_surfaces)]
    df = results_df

    if penalized:

        df['m penalty'] = df['stzd m'] - df['target m z']
        df.loc[df['stzd m'] < df['target m z'], 'm penalty'] = 0
        df[f'diff from target'] = np.sqrt((df[f'target r z'].to_numpy().astype(float)
                                                             - df[f'stzd r'].to_numpy().astype(float)) ** 2) + df[
                                                        'm penalty'].to_numpy().astype(float)

        for ppr in df['PrimerPairReporter'].unique():
            df.loc[df['PrimerPairReporter'] == ppr, f'best diff'] = [best_points.loc[
                                                                         best_points[
                                                                             'PrimerPairReporter'] == ppr,
                                                                         f'min error from targ comb']] * len(
                df.loc[df['PrimerPairReporter'] == ppr])
        if diff_from_target_only:
            df['combined regret'] = df[f'diff from target']
        else:
            df['combined regret'] = df[f'diff from target'] - df[f'best diff']
        regret_names.append('combined regret')

        for j, param in enumerate(params):

            df[f'diff from target {param}'] = ((df[f'stzd {param}'].to_numpy() - df[f'target {param} z'].to_numpy()) ** 2) ** (0.5)

            for ppr in df['PrimerPairReporter'].unique():
                df.loc[df['PrimerPairReporter'] == ppr, f'best diff {param}'] = [best_points.loc[
                                                                                     best_points[
                                                                                         'PrimerPairReporter'] == ppr, f'error from targ comb {param}']] * len(
                    df.loc[df['PrimerPairReporter'] == ppr])
            if diff_from_target_only:
                df[f'regret {param}'] = df[f'diff from target {param}']
            else:
                df[f'regret {param}'] = df[f'diff from target {param}'] - df[f'best diff {param}']
            regret_names.append(f'regret {param}')

    else:

        if 'r' in params and 'm' in params:


            df[f'diff from target'] = ((df[f'stzd r'].to_numpy() - df['target r z'].to_numpy()) ** 2
                                       + (df['stzd m'].to_numpy() - df['target m z'].to_numpy()) ** 2) ** (0.5)
            for ppr in df['PrimerPairReporter'].unique():
                df.loc[df['PrimerPairReporter'] == ppr, f'best diff'] = [best_points.loc[
                                                                             best_points[
                                                                                 'PrimerPairReporter'] == ppr,
                                                                             f'min error from targ comb']] * len(
                    df.loc[df['PrimerPairReporter'] == ppr])
            if diff_from_target_only:
                df['combined regret'] = df[f'diff from target']
            else:
                df['combined regret'] = df[f'diff from target'] - df[f'best diff']
            regret_names.append('combined regret')


        for j, param in enumerate(params):

            if stzd:
                df[f'diff from target {param}'] = ((df[f'stzd {param}'].to_numpy() - df[f'target {param} z'].to_numpy()) ** 2) ** (0.5)

                for ppr in df['PrimerPairReporter'].unique():
                    df.loc[df['PrimerPairReporter'] == ppr, f'best diff {param}'] = [best_points.loc[
                                                                                         best_points[
                                                                                             'PrimerPairReporter'] == ppr, f'min error from targ {param}']] * len(
                        df.loc[df['PrimerPairReporter'] == ppr])
                if diff_from_target_only:
                    df[f'regret {param}'] = df[f'diff from target {param}']
                else:
                    df[f'regret {param}'] = df[f'diff from target {param}'] - df[f'best diff {param}']
                regret_names.append(f'regret {param}')
            else:
                df[f'diff from target {param}'] = ((df[f'{param}'].to_numpy() - df[f'target {param}'].to_numpy()) ** 2) ** (0.5)
                for ppr in df['PrimerPairReporter'].unique():
                    df.loc[df['PrimerPairReporter'] == ppr, f'best diff {param}'] = [best_points.loc[
                                                                                         best_points[
                                                                                             'PrimerPairReporter'] == ppr, f'min error from targ {param}']] * len(
                        df.loc[df['PrimerPairReporter'] == ppr])
                if diff_from_target_only:
                    df[f'regret {param}'] = df[f'diff from target {param}']
                else:
                    df[f'regret {param}'] = df[f'diff from target {param}'] - df[f'best diff {param}']
                regret_names.append(f'regret {param}')
    df = df.astype({name: float for name in regret_names})
    df = df.groupby(['iteration', 'model', 'seed', 'PrimerPairReporter']).mean().reset_index()

    return df


def calculate_min_regret(df, params):
    """calculate the cumulative minimum regret and cumulative sum of regret"""

    if 'r' in params and 'm' in params:
        regret_names = ['combined regret', 'regret r', 'regret m']
    else:
        regret_names = [f'regret {param}' for param in params]

    regret_df = copy(df)
    regret_df = regret_df.sort_values('iteration')

    # calculate the cumulative minimum regret and cumulative sum of regret

    for model in df['model'].unique():
        for seed in df['seed'].unique():
            for ppr in df['PrimerPairReporter'].unique():
                for regret_name in regret_names:
                    regret_df.loc[
                        (regret_df['model'] == model) & (regret_df['seed'] == seed) & (
                                    regret_df['PrimerPairReporter'] == ppr), f'{regret_name} cummin'] = \
                        regret_df.loc[(regret_df['model'] == model) & (regret_df['seed'] == seed) & (
                                regret_df['PrimerPairReporter'] == ppr), regret_name].cummin()
                    regret_df.loc[
                        (regret_df['model'] == model) & (regret_df['seed'] == seed) & (
                                    regret_df['PrimerPairReporter'] == ppr), f'{regret_name} cumsum'] = \
                        regret_df.loc[(regret_df['model'] == model) & (regret_df['seed'] == seed) & (
                                regret_df['PrimerPairReporter'] == ppr), f'{regret_name} cummin'].cumsum()

    # calculate the mean, min, and max of the cumulative minimum regret and cumulative sum of regret across all seeds

    df4 = regret_df.groupby(['model', 'iteration', 'PrimerPairReporter']).mean().reset_index()
    df7 = regret_df.groupby(['model', 'iteration', 'PrimerPairReporter']).median().reset_index()
    df5 = regret_df.groupby(['model', 'iteration', 'PrimerPairReporter']).min().reset_index()
    df6 = regret_df.groupby(['model', 'iteration', 'PrimerPairReporter']).max().reset_index()

    # create new dataframe which combines these metrics

    metrics = ['mean', 'min', 'max', 'median']
    dfs = [df4, df5, df6, df7]
    for i in range(len(dfs)):
        for col in [f'{regret_name} {met}' for regret_name in regret_names for met in ['cummin', 'cumsum']]:
            dfs[i][f'{metrics[i]} {col}'] = dfs[i][col]
        dfs[i] = dfs[i].drop(
            columns=[f'{regret_name} {met}' for regret_name in regret_names for met in ['cummin', 'cumsum']]
                    + ['seed', 'index', 'initial_surface'] + regret_names)

    regret_df = dfs[0].merge(dfs[1], on=['model', 'iteration', 'PrimerPairReporter'])
    regret_df = regret_df.merge(dfs[2], on=['model', 'iteration', 'PrimerPairReporter'])
    regret_df = regret_df.merge(dfs[3], on=['model', 'iteration', 'PrimerPairReporter'])
    return regret_df


def plot_regret_per_surface(regret_df, results_df, params, metric='cumsum', plot_minmax=True, regret_names=None, save=False, test_name=None):
    """plot the regret per surface for each model"""
    colors = get_colors()

    if regret_names is None:
        if len(params) == 2:
            regret_names = ['combined regret', 'regret r', 'regret m']
        else:
            regret_names = [f'regret {param}' for param in params]

    if len(regret_names) == 1:
        ncols = 2
    else:
        ncols = len(regret_names)
    fig, axs = plt.subplots(nrows=len(regret_df['PrimerPairReporter'].unique()), ncols=ncols,
                            figsize=(4*ncols, 4 * len(regret_df['PrimerPairReporter'].unique())))

    for j, ppr in enumerate(regret_df['PrimerPairReporter'].unique()):
        # print(ppr)
        for i, regret_name in enumerate(regret_names):
            for model in regret_df['model'].unique():
                df = regret_df[(regret_df['model'] == model) & (regret_df['PrimerPairReporter'] == ppr)]
                axs[j, i].plot(df['iteration'], df[f'mean {regret_name} {metric}'], label=model, color=colors[model])
                if plot_minmax:
                    axs[j, i].fill_between(df['iteration'], df[f'min {regret_name} {metric}'], df[f'max {regret_name} {metric}'],
                                 alpha=0.2, color=colors[model])
                count_df = results_df[(results_df['PrimerPairReporter'] == ppr)][['GC', 'BP']].drop_duplicates()
                #                 axs[k, j].set_title(f'Regret {param} {ppr}\nno. points={len(df)}')
                axs[j, i].set_title(f'{regret_name} \nfor {ppr}\nno. points={len(count_df)}')
            axs[j, i].legend()
    if metric == 'cumsum':
        plt.suptitle('Cumulative Regret')
    elif metric == 'cummin':
        plt.suptitle('Cumulative Minimum Regret')
    plt.tight_layout()
    plt.savefig(f'plots/regret_per_surface_{test_name}.png')
    plt.show()


def plot_regret_per_surface_one_plot(regret_df, results_df, params, metric='cumsum', plot_minmax=True, regret_names=None, save=False, test_name=None):
    """plot the regret per surface for each model"""
    colors = get_colors()

    if regret_names is None:
        if len(params) == 2:
            regret_names = ['combined regret', 'regret r', 'regret m']
        else:
            regret_names = [f'regret {param}' for param in params]

    if len(regret_names) == 1:
        ncols = 2
    else:
        ncols = len(regret_names)
    fig, axs = plt.subplots(nrows=1, ncols=ncols,
                            figsize=(4*ncols, 4))

    for j, ppr in enumerate(regret_df['PrimerPairReporter'].unique()):
        # print(ppr)
        for i, regret_name in enumerate(regret_names):
            for model in regret_df['model'].unique():
                df = regret_df[(regret_df['model'] == model) & (regret_df['PrimerPairReporter'] == ppr)]
                axs[i].plot(df['iteration'], df[f'mean {regret_name} {metric}'], label=model, color=colors[model], alpha=0.5)
                if plot_minmax:
                    axs[i].fill_between(df['iteration'], df[f'min {regret_name} {metric}'], df[f'max {regret_name} {metric}'],
                                 alpha=0.2, color=colors[model])
                count_df = results_df[(results_df['PrimerPairReporter'] == ppr)][['GC', 'BP']].drop_duplicates()
                #                 axs[k, j].set_title(f'Regret {param} {ppr}\nno. points={len(df)}')
                axs[i].set_title(f'{regret_name} \nfor {ppr}\nno. points={len(count_df)}')
        # axs[i].legend()
    if metric == 'cumsum':
        plt.suptitle('Cumulative Regret')
    elif metric == 'cummin':
        plt.suptitle('Cumulative Minimum Regret')
    plt.tight_layout()
    plt.savefig(f'plots/regret_per_surface_{test_name}_one_plot.png')
    plt.show()


def plot_regret_per_surface_diff_layout(regret_df, results_df, params, metric='cumsum', plot_minmax=True,
                                        save=False, test_name=None, ncols=4):
    """plot the regret per surface for each model"""
    colors = get_colors()

    regret_name = 'combined regret'

    n_rows = int(np.ceil(len(regret_df['PrimerPairReporter'].unique())/ncols))

    fig, axs = plt.subplots(nrows=n_rows, ncols=ncols,
                            figsize=(4*ncols, 4 * (len(regret_df['PrimerPairReporter'].unique())/ncols)))
    axs = axs.flatten()
    for j, ppr in enumerate(regret_df['PrimerPairReporter'].unique()):
        # print(ppr)

        for model in regret_df['model'].unique():
            df = regret_df[(regret_df['model'] == model) & (regret_df['PrimerPairReporter'] == ppr)]
            axs[j].plot(df['iteration'], df[f'mean {regret_name} {metric}'], label=model, color=colors[model])
            if plot_minmax:
                axs[j].fill_between(df['iteration'], df[f'min {regret_name} {metric}'], df[f'max {regret_name} {metric}'],
                             alpha=0.2, color=colors[model])
            count_df = results_df[(results_df['PrimerPairReporter'] == ppr)][['GC', 'BP']].drop_duplicates()
            #                 axs[k, j].set_title(f'Regret {param} {ppr}\nno. points={len(df)}')
            axs[j].set_title(f'{ppr}\nno. points={len(count_df)}')
        axs[j].legend()
    if metric == 'cumsum':
        plt.suptitle('Cumulative Regret')
    elif metric == 'cummin':
        plt.suptitle('Cumulative Minimum Regret')
    plt.tight_layout()
    plt.savefig(f'plots/regret_per_surface_{test_name}.png')
    plt.show()

def plot_regret_all(regret_df, params, metric='cumsum', plot_minmax=True, save=False):


    colors = get_colors()

    if len(params) == 2:
        ncols = 3
        regret_names = ['combined regret', 'regret r', 'regret m']
    else:
        ncols = 2
        regret_names = [f'regret {param}' for param in params]
    fig, axs = plt.subplots(nrows=1, ncols=ncols,
                            figsize=(3 * ncols, 3))
    ax = axs.flatten()

    i_max = regret_df['iteration'].max()

    for ppr in regret_df['PrimerPairReporter'].unique():

        ppr_imax = regret_df.loc[regret_df['PrimerPairReporter'] == ppr, 'iteration'].to_numpy().max()
        if ppr_imax == i_max:
            pass
        else:
            extra_rows = pd.concat([regret_df.loc[(regret_df['PrimerPairReporter'] == ppr)
                                            & (regret_df['iteration'] == ppr_imax)]] * int((i_max - ppr_imax)))
            extra_rows = extra_rows.sort_values('model')
            if ppr_imax < (i_max - 1):
                extra_rows['iteration'] = np.arange(ppr_imax + 1, i_max + 1).tolist() * int(len(regret_df['model'].unique()))
                regret_df = regret_df.append(extra_rows)
            elif ppr_imax == (i_max - 1):
                extra_rows['iteration'] = i_max
                regret_df = regret_df.append(extra_rows)
            else:
                pass

    regret_df = regret_df.groupby(['model', 'iteration']).mean().reset_index()
    labels = {'mo_indi':'MOGP', 'avg':'AvgGP', 'lmc': 'LMC', 'lvm':'LVMOGP'}
    for i, regret_name in enumerate(regret_names):
        for model in regret_df['model'].unique():
            df = regret_df[(regret_df['model'] == model)].sort_values('iteration')
            ax[i].plot(df['iteration'], df[f'mean {regret_name} {metric}'], label=labels[model],
                           color=colors[model])
            if plot_minmax:
                ax[i].fill_between(df['iteration'], df[f'min {regret_name} {metric}'],
                                       df[f'max {regret_name} {metric}'],
                                       alpha=0.2, color=colors[model])
            ax[i].set_title(f'{regret_name} \nfor all surfaces')
        ax[i].legend()
    if metric == 'cumsum':
        plt.suptitle('Cumulative Regret')
    elif metric == 'cummin':
        plt.suptitle('Cumulative Minimum Regret')
    plt.tight_layout()

    if save:
        plt.savefig('plots/cumulative_regret.png', dpi=1000)
    plt.show()


def get_cum_regret_table(regret_df1, regret_name):
    """winning model in this case is the one with the lowest final cumulative regret"""
    df = regret_df1
    win_df = pd.DataFrame(columns=['PrimerPairReporter'] + list(df['model'].unique()))
    win_df['PrimerPairReporter'] = df['PrimerPairReporter'].unique()

    for model in df['model'].unique():
        for seed in df['seed'].unique():
            for ppr in df['PrimerPairReporter'].unique():
                df.loc[
                    (df['model'] == model) & (df['seed'] == seed) & (
                            df['PrimerPairReporter'] == ppr), f'{regret_name} cummin'] = \
                    df.loc[(df['model'] == model) & (df['seed'] == seed) & (
                            df['PrimerPairReporter'] == ppr), regret_name].cummin()
                df.loc[
                    (df['model'] == model) & (df['seed'] == seed) & (
                            df['PrimerPairReporter'] == ppr), f'{regret_name} cumsum'] = \
                    df.loc[(df['model'] == model) & (df['seed'] == seed) & (
                            df['PrimerPairReporter'] == ppr), f'{regret_name} cummin'].cumsum()

    for ppr in df['PrimerPairReporter'].unique():
        winners = []
        for seed in range(1, 21):
            temp_df = df[(df['PrimerPairReporter'] == ppr) & (df['seed'] == seed)]
            if len(temp_df) < 1:
                winner = None
            else:
                zero_regret_dict = {model: None for model in temp_df['model'].unique()}

                for model in temp_df['model'].unique():
                    zero_regret_dict[model] = temp_df.loc[(temp_df['model'] == model), f'{regret_name} cumsum'].max()
                minval = min(zero_regret_dict.values())
                winner = [k for k, v in zero_regret_dict.items() if v == minval]
                winners.append(winner[:])
            flat_winners_list = [item for sublist in winners for item in sublist]
            for model in temp_df['model'].unique():
                win_df.loc[win_df['PrimerPairReporter'] == ppr, model] = flat_winners_list.count(model)

    win_df_totals = pd.DataFrame(win_df.sum(axis=0)).T[[mod for mod in df['model'].unique()]]
    return win_df, win_df_totals





