
from pyod.models.pca import PCA
from pyod.models.abod import ABOD
from pyod.models.loci import LOCI
from pyod.models.lof import LOF
from pyod.models.auto_encoder_torch import AutoEncoder


class PyodDetector:
    def __init__(self, algorithm, **kwargs):
        if algorithm == 'PCA':
            self.detector = PCA(**kwargs)
        if algorithm == 'pyod_AE':
            self.detector = AutoEncoder(**kwargs)
        else:
            raise ValueError(algorithm)

    def fit(self, *args, **kwargs):
        self.detector = self.detector.fit(*args, **kwargs)
        return self

    def score_samples(self, *args, **kwargs):
        return -self.detector.decision_function(*args, **kwargs)
