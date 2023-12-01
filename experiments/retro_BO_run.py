
import pathlib as pl
import os
import warnings
import tensorflow as tf
warnings.simplefilter("ignore")
from retro_BO import RetroBO
from test_case_setup import setup_problem
import sys

# ensure tensorflow doesn't use GPU or too many CPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.threading.set_intra_op_parallelism_threads(5)
tf.config.threading.set_inter_op_parallelism_threads(5)

run_args =['retro_BO_run.py', '19', '0', '0', '5', '0'] # sys.argv[:]  # get the arguments which determine the test type. Can also input these manually using a list
# e.g. ['retro_BO_run.py', '19', '0', '0', '5', '0']#

run_args[2] = '2'
random_if_none = True
seed = int(int(run_args[1]))


if int(run_args[1]) % 2 == 0:
    run_args[2] = '2'
    seed = int(int(run_args[1]) / 2)

else:
    run_args[2] = '0'
    seed = int((int(run_args[1])-1) / 2)

# # set seed
# seed = int(run_args[1])

run_args = [0] + run_args
start_point, params, initial_surfaces, learning_surfaces, drop_surfaces, test_name = setup_problem(run_args)

print('run_args:', run_args, 'start point: ', start_point, 'params:', params, 'test name:', test_name, 'seed:', seed)

# set up the problem
dims = ['BP', 'GC']
latent_dims = ['PrimerPairReporter']
model_names = ['mo_indi', 'lmc', 'avg', 'lvm']  # which models to run. Should be out of: 'mo_indi', 'lmc', 'avg', 'lvm'
coregion_rank = 2
if start_point == 'centre':
    pct_train = 0.98
    warm_start = True
    one_of_each = True
else:
    pct_train = 0.0
    warm_start = False
    one_of_each = False

n_restarts = 10

if not warm_start:
    test_name = f'{test_name}_0_point_start'
one_of_each = True

log_transform = False
log_t = 'No_Transform'

# create retroBO class
retroBO = RetroBO(dims, latent_dims, coregion_rank, params, model_names, log_t, ei='penalized',
                  random_if_none=random_if_none)

path = pl.Path(os.getcwd()).parent

# split the data
train_ps, test_ps = retroBO.divide_data(file_name='ADVI_ParameterSets_220528.pkl', seed=seed, n_train=None,
                                        pct_train=pct_train,
                                        initial_surfaces=initial_surfaces,
                                        warm_start=warm_start, one_of_each=one_of_each, drop_surfaces=drop_surfaces,
                                        start_point=start_point, log_transform=log_transform)
retroBO.make_data_splits()

# run the bayes opt iteration
results_df, predictions = retroBO.run_BO(max_iteration=53, n_restarts=n_restarts, save=True, seed=seed,
                                         test_type_name=test_name)



