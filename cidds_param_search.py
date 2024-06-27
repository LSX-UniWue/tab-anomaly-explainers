
from sklearn.model_selection import ParameterGrid

from cidds_detectors import detect_anomalies


def get_param_grid(algorithm):
    if algorithm == 'IsolationForest':
        return ParameterGrid({'n_estimators': [32],
                              'max_samples': [0.8],
                              'max_features': [0.8],
                              'bootstrap': [0],
                              'n_jobs': [-1],
                              'random_state': [0]})

    elif algorithm == 'OneClassSVM':
        return ParameterGrid({"kernel": ['rbf'],
                              'gamma': [1e1],
                              'tol': [1e-3],
                              'nu': [0.2],
                              'shrinking': [1],
                              'cache_size': [500],
                              'max_iter': [-1],
                              'seed': [0]})

    elif algorithm == 'Autoencoder':
        return ParameterGrid({'cpus': [8], 'n_layers': [3], 'n_bottleneck': [32], 'epochs': [2], 'batch_size': [2048],
                              'learning_rate': [1e-2], 'verbose': [2], 'device': ['cuda'], 'seed': list(range(17, 20))})  # best

    else:
        raise ValueError(f"Variable algorithm was: {algorithm}")


if __name__ == '__main__':

    # algorithm = 'Autoencoder'  # One of ['Autoencoder', 'OneClassSVM', 'IsolationForest']
    # evaluation_save_path = 'outputs/logs/cidds'
    # job_name_template = 'local_{}'  # saves model at f'./outputs/models/cidds/{algorithm}/{job_name}', None to not save
    # for i, param_dict in enumerate(get_param_grid(algorithm=algorithm)):
    #     job_name = job_name_template.format(str(i))
    #     seed = param_dict['random_state'] if 'random_state' in param_dict else param_dict['seed']
    #     detect_anomalies(algorithm=algorithm,
    #                      params=param_dict,
    #                      seed=seed,
    #                      evaluation_save_path=evaluation_save_path,
    #                      job_name=job_name)

    algorithm = 'Autoencoder'
    evaluation_save_path = 'outputs/logs/cidds/rashomon'
    job_name_template = 'rashomon_{}'  # saves model at f'./outputs/models/cidds/{algorithm}/{job_name}', None to not save
    for i, param_dict in enumerate(get_param_grid(algorithm=algorithm)):
        seed = param_dict['random_state'] if 'random_state' in param_dict else param_dict['seed']
        job_name = job_name_template.format(str(seed))
        detect_anomalies(algorithm=algorithm,
                         params=param_dict,
                         seed=seed,
                         evaluation_save_path=evaluation_save_path,
                         job_name=job_name)
