
import pickle
from pathlib import Path

from anomaly_detection.autoencoder_torch import Autoencoder


def load_best_detector(model, train_path, test_path, model_folder='./outputs/models/erp_fraud/'):
    if model == 'AE':
        if 'mas_scm_data_fraud2' in train_path:
            detector = Autoencoder(**pickle.load(open(Path('./outputs/models/erp_mas/best_params_mas.p'), 'rb')))
            detector = detector.load(Path('./outputs/models/erp_mas/') / 'AE_mas1', only_model=False)
        elif 'mas_scm_data_fraud3' in train_path:
            detector = Autoencoder(**pickle.load(open(Path('./outputs/models/erp_mas/best_params_mas.p'), 'rb')))
            detector = detector.load(Path('./outputs/models/erp_mas/') / 'AE_mas2', only_model=False)
        elif 'fraud_2' in train_path:
            detector = Autoencoder(**pickle.load(open(Path('./outputs/models/erp_mas/best_params_erpsim1.p'), 'rb')))
            detector = detector.load(Path('./outputs/models/erp_mas/') / 'AE_erpsim1', only_model=False)
        elif 'fraud_3' in train_path:
            detector = Autoencoder(**pickle.load(open(Path('./outputs/models/erp_mas/best_params_erpsim2.p'), 'rb')))
            detector = detector.load(Path('./outputs/models/erp_mas/') / 'AE_erpsim2', only_model=False)
        elif 'normal_1' in train_path:
            detector = Autoencoder(**pickle.load(open(Path(model_folder) / 'best_params/AE_session1.p', 'rb')))
            detector = detector.load(Path(model_folder) / 'AE_session1_torch', only_model=True)
        elif 'normal_2' in train_path:
            detector = Autoencoder(**pickle.load(open(Path(model_folder) / 'best_params/AE_session2.p', 'rb')))
            detector = detector.load(Path(model_folder) / 'AE_session2_torch', only_model=True)
        else:
            raise ValueError("Unknown train and test dataset combination.")
    elif model == 'IF':
        from sklearn.ensemble import IsolationForest
        detector = IsolationForest(**pickle.load(open(Path(model_folder) / 'best_params/IF.p', 'rb')))
    elif model == 'PCA':
        from anomaly_detection.pyod_wrapper import PyodDetector
        detector = PyodDetector(**pickle.load(open(Path(model_folder) / 'best_params/PCA.p', 'rb')))
    elif model == 'OCSVM':
        import joblib
        from sklearn.svm import OneClassSVM
        from dask_ml.wrappers import ParallelPostFit
        detector = joblib.load(f'./outputs/models/erp_fraud/OC_SVM_session2.pkl')
        # detector = ParallelPostFit(estimator=DaskOCSVM(detector))
    else:
        raise ValueError(f"Expected 'model' to be one of ['OCSVM', 'AE', 'PCA'], but was {model}")

    return detector


class DaskOCSVM:
    """Small wrapper to trick dask_ml into parallelizing anomaly detection methods"""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.score_samples(X)
