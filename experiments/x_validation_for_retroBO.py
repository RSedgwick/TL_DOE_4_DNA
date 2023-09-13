
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

# ensure tensorflow doesn't use GPU or too many CPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.threading.set_intra_op_parallelism_threads(5)
tf.config.threading.set_inter_op_parallelism_threads(5)

run_args = sys.argv[:]  # get the arguments which determine the test type. Can also input these manually using a list
print(run_args)
seed = int(run_args[1])

pct_train = int(run_args[2]) / 100
start_point, params, initial_surfaces, learning_surfaces, drop_surfaces, test_name = setup_problem(run_args)

log_transform = False

dims = ['BP', 'GC']
latent_dims = ['PrimerPairReporter']
model_names = ['mo_indi', 'lmc', 'avg', 'lvm'] # which models to run. Should be out of: 'mo_indi', 'lmc', 'avg', 'lvm'
coregion_rank = 2
warm_start = True
one_of_each = False
n_restarts = 10

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
log_t = 'No_Transform'

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
