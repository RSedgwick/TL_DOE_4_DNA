
import warnings
warnings.simplefilter("ignore")
from x_validation_functions import CrossValidation
import tensorflow as tf
import pathlib as pl
import os
import pickle
from test_case_setup import setup_problem
import sys
import matplotlib.pyplot as plt
import argparse

# ensure tensorflow doesn't use GPU or too many CPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tf.config.threading.set_intra_op_parallelism_threads(5)
# tf.config.threading.set_inter_op_parallelism_threads(5)



def main(args):

    # run_args = ['retro_BO_run.py', '19', '50', '0', '5', '0']  #sys.argv[:]  # get the arguments which determine the test type. Can also input these manually using a list
    # print(run_args)

    seed = args.seed
    pct_train = args.pct_train
    start_point = args.start_point

    # assign the param name which will go in the run name and the params to optimise
    params_name = args.params_name

    if args.params_name == 'both':
        params = ['r', 'm']
    else:
        params = [args.params_name]

    all_surfaces = ['FP004-RP004-Probe', 'FP001-RP001x-EvaGreen', 'FP002-RP002x-EvaGreen',
                     'FP001-RP001x-Probe', 'FP005-FP001-Probe', 'RP001x-FP002-Probe',
                     'RP002x-FP005-Probe', 'FP005-FP004-EvaGreen', 'RP002x-FP002-EvaGreen',
                     'FP001-RP004-EvaGreen', 'FP002-RP004-EvaGreen', 'FP004-FP005-Probe',
                     'RP008x-FP005-Probe', 'FP005-FP001-EvaGreen', 'RP002x-FP004-EvaGreen',
                     'RP008x-FP001-EvaGreen', 'FP001-RP001-Probe', 'FP002-RP002-Probe', 'FP004-RP004-EvaGreen',
                     'FP001-RP001-EvaGreen', 'FP002-RP002-EvaGreen', 'FP001-RP005-Probe',
                     'FP005-RP005-Probe', 'FP002-RP006-Probe', 'FP006-RP006-Probe',
                     'FP003-RP008-Probe', 'FP002-RP002x-Probe', 'FP004-RP004x-Probe',
                     'FP004-RP004x-EvaGreen', 'FP003-RP008-EvaGreen', 'FP003-RP008x-EvaGreen',
                     'FP057.1.0-RP003x-EvaGreen', 'FP003-RP003-Probe']
    
    drop_surfaces=None
    learning_surfaces=['FP057.1.0-RP003x-Probe']


    initial_surfaces = {surface:'all' for surface in all_surfaces}
    
    log_transform = False

    dims = ['BP', 'GC']
    latent_dims = ['PrimerPairReporter']
    model_names = ['lmc','lvm'] # which models to run. Should be out of: 'mo_indi', 'lmc', 'avg', 'lvm'
    coregion_rank = 10
    warm_start = True
    one_of_each = False
    n_restarts = 1

    # create cross validation class
    Xvalid = CrossValidation(dims, latent_dims, coregion_rank, params, model_names)

    # divide the data
    train_ps, test_ps = Xvalid.divide_data(file_name='ADVI_ParameterSets_220528.pkl', seed=seed, n_train=None,
                                        pct_train=pct_train,
                                        initial_surfaces=initial_surfaces,
                                        warm_start=warm_start, one_of_each=one_of_each, drop_surfaces=drop_surfaces,
                                        start_point=start_point, log_transform=log_transform)

    Xvalid.print_train_test_split()

    # fit the models and calculate performance metrics
    Xvalid.fit_models(n_restarts=n_restarts, train_ps=Xvalid.train_ps)
    results_dfs, test_dfs = Xvalid.get_metrics()

    print('got metrics, saving results')

    # save the results

    path = pl.Path(os.getcwd()).parent.parent/f'Results/Xvalid_for_RetroBO/restarts_{n_restarts}_3008'

    for param in params:

        results_dfs[param]['seed'] = seed
        results_dfs[param]['n_train'] = Xvalid.n_train
        results_dfs[param]['pct_train'] = Xvalid.pct_train
        path_df = path / f'results_{test_name}_{int(pct_train*100)}_{seed}_{param}.pkl'

        with open(path_df, "wb") as file:
            pickle.dump(results_dfs[param], file)

        for ps_name in ['train', 'test']:
            test_dfs[param][ps_name]['seed'] = seed
            test_dfs[param][ps_name]['n_train'] = Xvalid.n_train
            test_dfs[param][ps_name]['pct_train'] = Xvalid.pct_train

            path_test_df = path / f'{ps_name}_df_{test_name}_{int(pct_train*100)}_{seed}_{param}.pkl'

            with open(path_test_df, "wb") as file:
                pickle.dump(test_dfs[param][ps_name], file)


if __name__ == "__main__":
    import os

    os.environ[
        "WANDB__SERVICE_WAIT"
    ] = "300"  # this is to prevent wandb from waiting for a response from the server


    argparser = argparse.ArgumentParser()
    
    argparser.add_argument(
        "--work_dir",
        "-wd",
        type=str,
        default="./",
        help="Set your working directory to the top level of the repo.",
    )

    argparser.add_argument(
    "--seed",
    "-s",
    type=int,
    default=19,
    help="Random seed.",
    )

    argparser.add_argument(
        "--pct_train",
        "-pt",
        type=float,
        default=0.5,
        help="Percentage of data to use for training.",
    )

    argparser.add_argument(
    "--start_point",
    "-sp",
    type=str,
    default='centre',
    help="First point for each of the models. Out of 'worst_point', 'centre', 'None'",
    )

    argparser.add_argument(
        "--params_name",
        "-pn",
        type=str,
        default='both',
        help="Which parameters to use. Out of 'both', 'r', 'm'.",
    )

    argparser.add_argument(
        "--surface_splits",
        "-ss",
        type=int,
        default=0,
        help="Indicator of how the surfaces should be split.",)

    args = argparser.parse_args()
    main(args)

