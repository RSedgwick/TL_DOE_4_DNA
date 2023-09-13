
def setup_problem(run_args):
    """function to set up the problem to be solved based on list of run args. Returns how the starting point should
    be chosen, which parameters to optimise (m, r, or both), the initial surfaces to be observed and how many
    datapoints on each, the surfaces to be learnt and the surfaces to be dropped. Also the name of the test case.
    :param run_args: list of run args
    :return: start_point, params, initial_surfaces, learning_surfaces,
    drop_surfaces, test_name"""

    if int(run_args[3]) == 1:
        start_point = 'worst_point'
        start_point_name = '_worst_point'
    elif int(run_args[3]) == 0:
        start_point = None
        start_point_name = ''
    else:
        start_point = 'centre'
        start_point_name = '_centre'

    if int(run_args[4]) == 0:
        params = ['r', 'm']
        param_name = 'both'
    elif int(run_args[4]) == 1:
        params = ['r']
        param_name = 'r'
    else:
        params = ['m']
        param_name = 'm'

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
                     'FP057.1.0-RP003x-EvaGreen', 'FP003-RP003-Probe', 'FP057.1.0-RP003x-Probe']

    out_targ = ['FP001-RP001-Probe', 'FP002-RP002-Probe', 'FP004-RP004-EvaGreen',
                'FP001-RP001-EvaGreen', 'FP002-RP002-EvaGreen', 'FP001-RP005-Probe',
                'FP005-RP005-Probe', 'FP002-RP006-Probe', 'FP006-RP006-Probe',
                'FP003-RP008-Probe', 'FP002-RP002x-Probe', 'FP004-RP004x-Probe',
                'FP004-RP004x-EvaGreen', 'FP003-RP008-EvaGreen', 'FP003-RP008x-EvaGreen',
                'FP057.1.0-RP003x-EvaGreen', 'FP003-RP003-Probe', 'FP057.1.0-RP003x-Probe']


    if int(run_args[5]) == 0:
        initial_surfaces = {'FP004-RP004-EvaGreen': 'all', 'FP002-RP002x-Probe': 'all'}

        learning_surfaces = ['FP004-RP004-Probe']
        surface_name = f'one_from_few_{learning_surfaces[0]}'

    if int(run_args[5]) == 1:
        surfs =  ['FP001-RP001-Probe', 'FP002-RP002-Probe', 'FP004-RP004-EvaGreen',
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
        initial_surfaces = {surface: 'all' for surface in surfs}

        learning_surfaces = ['FP004-RP004-Probe']
        surface_name = f'one_from_many_{learning_surfaces[0]}'

    if int(run_args[5]) == 2:
        initial_surfaces = {'FP004-RP004-EvaGreen': 'all', 'FP002-RP002x-Probe': 'all'}

        learning_surfaces = list(set(all_surfaces) - set(list(initial_surfaces.keys())) - set(out_targ))
        surface_name = f'many'

    if int(run_args[5]) == 3:
        initial_surfaces = {'FP004-RP004-EvaGreen': 'all', 'FP002-RP002x-Probe': 'all'}

        learning_surfaces = ['FP002-RP002x-EvaGreen']
        surface_name = f'one_from_few_{learning_surfaces[0]}'

    if int(run_args[5]) == 4:
        surfs = ['FP001-RP001-Probe', 'FP002-RP002-Probe', 'FP004-RP004-EvaGreen',
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
        initial_surfaces = {surface: 'all' for surface in surfs}
        learning_surfaces = ['FP002-RP002x-EvaGreen']
        surface_name = f'one_from_many_{learning_surfaces[0]}'

    if int(run_args[5]) > 4:

        all_surfs = ['FP001-RP001-Probe', 'FP002-RP002-Probe', 'FP004-RP004-EvaGreen',
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
                 'RP008x-FP001-EvaGreen', 'FP002-RP002x-EvaGreen']

        surfs = ['FP004-RP004-Probe', 'FP001-RP001x-EvaGreen', 'FP002-RP002x-EvaGreen',
                'FP001-RP001x-Probe', 'FP005-FP001-Probe', 'RP001x-FP002-Probe',
                'RP002x-FP005-Probe', 'FP005-FP004-EvaGreen', 'RP002x-FP002-EvaGreen',
                'FP001-RP004-EvaGreen', 'FP002-RP004-EvaGreen', 'FP004-FP005-Probe',
                'RP008x-FP005-Probe', 'FP005-FP001-EvaGreen', 'RP002x-FP004-EvaGreen',
                'RP008x-FP001-EvaGreen']

        learning_surfaces = [surfs[int(run_args[5]) - 5]]
        all_surfs.remove(learning_surfaces[0])
        initial_surfaces = {surface: 'all' for surface in all_surfs}
        surface_name = f'one_from_many_{learning_surfaces[0]}'

    drop_surfaces = set(all_surfaces) - set(learning_surfaces) - set(list(initial_surfaces.keys()))
    drop_surfaces = list(drop_surfaces)
    test_name = f'{surface_name}_{param_name}{start_point_name}'

    return start_point, params, initial_surfaces, learning_surfaces, drop_surfaces, test_name
