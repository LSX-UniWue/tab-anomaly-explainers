from sklearn.model_selection import ParameterGrid

from erp_detectors import detect_anomalies


def get_param_grid(algorithm, seed):
    if algorithm == 'IsolationForest':
        # return ParameterGrid({'n_estimators': [2 ** n for n in range(4, 8)],  # param search
        #                       'max_samples': [0.4, 0.6, 0.8], 'max_features': [0.4, 0.6, 0.8],
        #                       'bootstrap': [0], 'n_jobs': [-1]})  # param search
        # return ParameterGrid({'n_estimators': [100], 'max_samples': [256], 'max_features': [1.0],
        #                       'bootstrap': [0], 'n_jobs': [-1], 'random_state': [seed]})  # pyod default
        # return ParameterGrid({'n_estimators': [128], 'max_samples': [0.8], 'max_features': [0.8],
        #                       'bootstrap': [0], 'n_jobs': [-1], 'random_state': [seed]})  # best fraud2 zscore
        return ParameterGrid({'n_estimators': [128], 'max_samples': [0.8], 'max_features': [0.8],
                              'bootstrap': [0], 'n_jobs': [-1], 'random_state': [seed]})  # best fraud3 zscore

    elif algorithm == 'Autoencoder':
        # return ParameterGrid({'cpus': [8], 'n_layers': [2], 'n_bottleneck': [32], 'epochs': [50],  # MAS runs best
        #                       'batch_size': [2048], 'learning_rate': [1e-2], 'shuffle': [True],
        #                       'verbose': [1], 'device': ['cuda'], 'save_path': ['outputs/models/erp_mas/AE_erpsim2']})
        # return ParameterGrid({'n_layers': [2, 3, 4], 'n_bottleneck': [8, 16, 32], 'learning_rate': [1e-2, 1e-3, 1e-4],
        #                       'epochs': [50], 'batch_size': [2048], 'cpus': [8], 'shuffle': [True], 'verbose': [1],
        #                       'device': ['cuda'], 'save_path': [None]})
        return ParameterGrid({'cpus': [8], 'n_layers': [2], 'n_bottleneck': [32], 'epochs': [50],  # Rashomon consistency
                              'batch_size': [2048], 'learning_rate': [1e-2], 'shuffle': [True],
                              'verbose': [1], 'device': ['cuda'],
                              'save_path': ['outputs/models/erp_fraud/AE_rashomon/AE_seed_{}']})
        # return ParameterGrid({'cpus': [8], 'n_layers': [2], 'n_bottleneck': [32], 'epochs': [50],  # param search best
        #                       'batch_size': [2048], 'learning_rate': [1e-2], 'shuffle': [True],
        #                       'verbose': [1], 'device': ['cuda'], 'save_path': [None]})

    elif algorithm == 'OneClassSVM':
        # return ParameterGrid({"kernel": ['rbf'], 'nu': [0.5],  # pyod default
        #                       'gamma': [1/36778],  # fraud_2.csv: 1/36778, fraud_3.csv: 1/37407
        #                       'tol': [1e-3], 'shrinking': [1], 'cache_size': [500], 'max_iter': [-1]})
        return ParameterGrid({"kernel": ['rbf'], 'nu': [0.05],  # best fraud2+3 buckets
                              'gamma': [1], 'tol': [1e-3], 'shrinking': [1], 'cache_size': [500], 'max_iter': [-1]})

    elif algorithm == 'PCA':
        return ParameterGrid({'n_components': [0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
                              'whiten': [True, False],
                              'random_state': [seed],
                              'weighted': [True, False],
                              'standardization': [False]})

    elif algorithm == 'pyod_AE':
        # https://pyod.readthedocs.io/en/latest/pyod.models.html#pyod.models.auto_encoder.AutoEncoder
        return [{'preprocessing': False,}]  # already processed TODO: their dataloader .item() still tries to normalize... needs mean=np.array([0]) on creations for .any() to fail

    else:
        raise ValueError(f"Variable algorithm was: {algorithm}")


if __name__ == '__main__':

    seeds = list(range(100))  # [0], list(range(5))
    numeric = 'buckets'  # One of ['log10', 'zscore', 'buckets', 'minmax', 'None']
    # One of ['Autoencoder', 'OneClassSVM', 'IsolationForest', 'LocalOutlierFactor', 'PCA', 'NALU', 'VAE', 'pyod_AE']
    algorithm = 'Autoencoder'
    out_template = f'{algorithm}_erp_fraud_{{}}'

    erpClassParams = {
                      # 'train_path': './data/erp_mas/normal/mas_scm_data_fraud3.csv',
                      'train_path': './data/erp_fraud/normal_2.csv',
                      # 'train_path': './data/erp_fraud/fraud_2.csv',
                      # 'train_path': './data/erp_fraud/fraud_3.csv',
                      # 'eval_path': None,
                      'eval_path': './data/erp_fraud/fraud_2.csv',
                      # 'test_path': './data/erp_fraud/fraud_2.csv',
                      'test_path': './data/erp_fraud/fraud_3.csv',
                      # 'test_path': './data/erp_mas/fraud/mas_scm_data_fraud3.csv',
                      'info_path': './data/erp_fraud/column_information.csv',  # column names and dtypes
                      'split_id': None,  # split index
                      }

    for seed in seeds:
        param_grid = get_param_grid(algorithm=algorithm, seed=seed)
        for j, params in enumerate(param_grid):
            detect_anomalies(algorithm=algorithm,
                             **erpClassParams,
                             experiment_name=out_template.format(str(seed) + '_' + str(j)),
                             categorical='onehot',
                             numeric=numeric,
                             numeric_nan_bucket=False,
                             params=params,
                             output_scores=True,
                             seed=seed)
