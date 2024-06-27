
import numpy as np


def _percent_to_int(k, n):
    assert 0. < k <= 1.
    return int(np.ceil(k*n))


def _top_k_feature_agreement(a, b, k=3):
    if k < 1:
        k = _percent_to_int(k, a.shape[0])
    _top_k_a = np.argsort(np.abs(a))[:, -k:][:, ::-1]  # [:, ::-1] to get ascending order
    _top_k_b = np.argsort(np.abs(b))[:, -k:][:, ::-1]

    agreements = np.zeros(a.shape[0])

    for i, (ka, kb) in enumerate(zip(_top_k_a, _top_k_b)):
        agreements[i] = len(set(ka).intersection(set(kb))) / k

    return agreements


def _top_k_sign_agreement(a, b, k=3):
    if k < 1:
        k = _percent_to_int(k, a.shape[0])
    _top_k_a = np.argsort(np.abs(a))[:, -k:][:, ::-1]  # [:, ::-1] to get ascending order
    _top_k_b = np.argsort(np.abs(b))[:, -k:][:, ::-1]
    signs = a*b
    def _sign_agreement(_ka, _kb, _s, _k):# _a, _b, _k):
        s = 0
        for idx in range(_k):
            if _ka[idx] in _kb and _s[_ka[idx]] >= 0:  # to check if top feature overlaps
                # _kbidx = np.argwhere(_kb == _ka[idx]).item()  # get idx of
                # if _a[_ka[idx]] * _b[_kbidx] > 0 or (_a[_ka[idx]] == _b[_kbidx] == 0):
                s += 1
        return s / _k
    agreements = np.zeros(a.shape[0])
    for i, (ka, kb, _s) in enumerate(zip(_top_k_a, _top_k_b, signs)):
        agreements[i] = _sign_agreement(ka, kb, _s,  k)

    return agreements


def _elementwise_eucl(a, b):
    return np.linalg.norm(a-b, axis=1)


def _k_rankings(x1, x2, k=0.25, select='top'):
    assert x1.shape[0] == x2.shape[0]
    n_features = x1.shape[1]
    n_datapoints = x1.shape[0]

    cutoff = int(n_features * k)

    r1 = np.argsort(x1)[:, -cutoff:]
    r2 = np.argsort(x2)[:, -cutoff:]

    similarities = _sets_windowed(r1, r2, window_size=3)

    return similarities


def _sets_windowed(a, b, window_size=3):
    n_datapoints, n_features = a.shape
    similarities = np.zeros((n_datapoints))
    pair_similarities = []

    for datapoint_index in range(len(a)):
        for w_start in range(0, n_features-window_size, 1):
            w_stop = w_start + window_size

            set_a = set(a[datapoint_index, w_start:w_stop])
            set_b = set(b[datapoint_index, w_start:w_stop])

            # Because distance measure: 1-similarity
            pair_similarities.append(1 - (len(set_a.intersection(set_b)) / window_size))

        similarities[datapoint_index] = np.mean(pair_similarities)
    return similarities


def _top_k_rankings(x1, x2, k=0.25):
    return _k_rankings(x1, x2, k=k, select='top')


def _k_euclid(x1, x2, k=0.25, select='top'):

    assert x1.shape[0] == x2.shape[0]
    n_features = x1.shape[1]
    cutoff = int(n_features * k)

    x1args = np.argsort(x1)[:, -cutoff:]
    x1 = x1[:, x1args]
    x2args = np.argsort(x2)[:, -cutoff:]
    x2 = x2[:, x2args]

    return _explanation_distance(x1, x2)


def _explanation_distance(expl1, expl2, reduction=np.mean, metric=_elementwise_eucl, shape=None, **kwargs):
    d = metric(expl1, expl2, **kwargs)
    return reduction(d)


def _top_k_eucl(x1, x2, k=0.25):
    return _k_euclid(x1, x2, k=k, select='top')
