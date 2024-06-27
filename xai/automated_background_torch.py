
import tqdm
from functools import partial

from scipy.optimize import minimize
import numpy as np
import torch
from torchmetrics.functional import structural_similarity_index_measure
import wandb


def optimize_input_gradient_descent(data_point, mask, predict_fn, threshold=None, previous_points=None, max_iter=10000,
                                    lr=0.001, proximity_weight=0.01, diversity_weight=0.01, diversity_radius=0.01,
                                    flip_channels=False, device='cpu', verbose=1, dist_type='l2', **dist_kwargs):
    """
    Optimizes given sample for given model function, while keeping certain values in data_point fixed.
    loss = model(y) + gamma * mean squared distance between optimized and original point (excluding fixed values)
    :param data_point:          numpy model input to optimize
    :param mask:                numpy binary mask noting which features should be optimized: 0 means optimize, 1 means keep
    :param threshold:           if not None, uses hinge loss with classification threshold: max(0, anomaly_score(x) - threshold)
    :param predict_fn:          function of< pytorch model to optimize loss for
    :param previous_points:     numpy array of previously optimized datapoints to keep away from during optimization
    :param max_iter:            int gradient descent updates
    :param lr:                  float gradient descent learning rate (Adam optimizer)
    :param proximity_weight:    float hyperparameter weighting distance to original point in loss function
    :param diversity_weight:    float hyperparameter weighting average distance to previously found points in loss function
    :return:                    numpy optimized data_point
    example:
    x_hat.append(optimize_input_gradient_descent(data_point=sample, mask=mask, kept_feature_idx=kept_idx,
                                                       model=model, max_iter=5000, lr=0.001, gamma=0.01))
    """
    if flip_channels:
        data_point = np.transpose(data_point, [0, 3, 1, 2])
        mask = np.transpose(mask, [0, 3, 1, 2])

    def dist_ssim(x1, x2, luminance_factor=0.01, contrast_factor=0.03):
        ssim, contrast, output_list = structural_similarity_index_measure(preds=x1,
                                                                          target=x2,
                                                                          k1=luminance_factor,
                                                                          k2=contrast_factor,
                                                                          return_contrast_sensitivity=True)
        luminance = ssim / contrast
        dssim = (1 - ssim) / 2  # structural dissimilarity
        return dssim, luminance, contrast

    def dist_norm(x1, x2, **kwargs):
        if 'ord' not in locals():
            ord = 2  # euclidean norm if not specified in kwargs
        return torch.linalg.norm(x1.flatten() - x2.flatten(), ord=ord)

    if previous_points is not None:
        previous_points = torch.from_numpy(previous_points.astype('float32')).to(device)
    orig_point = torch.from_numpy(data_point.astype('float32')).to(device)
    data_point = torch.from_numpy(data_point.astype('float32')).to(device).requires_grad_()
    proximity_weight = torch.Tensor([proximity_weight]).to(device)
    diversity_weight = torch.Tensor([diversity_weight]).to(device)
    diversity_radius = torch.Tensor([diversity_radius]).to(device)
    if threshold is not None:
        threshold = torch.Tensor([threshold]).to(device)

    if dist_type == 'norm':
        dist = dist_norm
    elif dist_type == 'ssim':
        dist = dist_ssim
    else:
        raise ValueError(f'Unknown dist_type: {dist_type}')

    # convert constants to tensors
    mask = torch.from_numpy(mask.astype(bool)).to(device)
    invert_mask = ~mask
    constrained_vals = orig_point * mask

    # optimization procedure
    optimizer = torch.optim.Adam(params=[data_point], lr=lr)

    zero_elem = torch.tensor(0)
    one_elem = torch.tensor(1)
    patience = 10
    delta = 0
    lowest_score = None
    patience_counter = 0
    with tqdm.tqdm(range(max_iter), disable=verbose < 1) as titer:
        for i in titer:
            # forward pass & mask fixed inputs
            yloss = predict_fn(data_point)
            if threshold is not None:  # hinge loss where everything < threshold is considered normal
                yloss = torch.max(zero_elem, yloss - threshold)

            # DICE distance loss for multiple data points: https://github.com/interpretml/DiCE
            # removed all but current point, since we only optimize one at the time
            if dist_type == 'ssim':
                proximity_loss, luminance, contrast = dist(orig_point, data_point, **dist_kwargs)
            else:
                proximity_loss = dist(orig_point, data_point, **dist_kwargs)

            # DICE diversity loss (avg-dist): https://github.com/interpretml/DiCE
            # only compare to currently optimized single datapoint since we only optimize one at the time
            diversity_loss = 0.0
            if previous_points is not None:
                # computing pairwise distance
                for i in range(previous_points.shape[0]):
                    if dist_type == 'ssim':
                        dist_i, _, _ = dist(torch.unsqueeze(previous_points[i], 0), data_point, **dist_kwargs)
                    else:
                        dist_i = dist(torch.unsqueeze(previous_points[i], 0), data_point, **dist_kwargs)

                    if dist_type == 'ssim':  # un-similarity in interval [0, 1]: simply invert and subtract radius
                        diversity_loss += torch.max(zero_elem, one_elem - dist_i - diversity_radius)
                    else:  # real distance, needs gating
                        diversity_loss += torch.max(zero_elem, -1 * dist_i + diversity_radius)

            # DICE loss for categorical features: https://github.com/interpretml/DiCE
            # regularization_loss = 0.0
            # for i in range(total_points.shape[0]):
            #     for v in encoded_categorical_feature_indexes:
            #         regularization_loss += torch.pow((torch.sum(stotal_points[i][v[0]:v[-1]+1]) - 1.0), 2)

            loss = yloss + \
                   (proximity_weight * proximity_loss) + \
                   (diversity_weight * diversity_loss)

            if i % 10 == 0:
                log_dict = {'optim_anomaly_loss': yloss,
                            'optim_proximity_loss': proximity_weight * proximity_loss,
                            'optim_proximity': proximity_loss,
                            'optim_diversity_loss': diversity_weight * diversity_loss,
                            'optim_diversity': diversity_loss,
                            'optim_loss': loss}
                if dist_type == 'ssim':
                    log_dict['optim_luminance'] = luminance
                    log_dict['optim_contrast'] = contrast
                wandb.log(log_dict)

            # early stopping
            if lowest_score is None or loss < lowest_score - delta:
                lowest_score = loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                wandb.log({'optim_anomaly_loss': yloss,
                           'optim_proximity_loss': proximity_weight * proximity_loss,
                           'optim_proximity': proximity_loss,
                           'optim_diversity_loss': -1 * diversity_weight * diversity_loss,
                           'optim_diversity': diversity_loss,
                           'optim_loss': loss})
                if flip_channels:
                    return np.transpose(data_point.cpu().detach().numpy(), [0, 2, 3, 1])
                else:
                    return data_point.cpu().detach().numpy()

            # Gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            data_point.data = data_point.data * invert_mask + constrained_vals  # Reset fixed inputs with mask

    if flip_channels:
        return np.transpose(data_point.cpu().detach().numpy(), [0, 2, 3, 1])  # optimized sample
    else:
        return data_point.cpu().detach().numpy()


def optimize_input_quasi_newton(data_point, kept_feature_idx, predict_fn, threshold=None, points_to_keep_distance=(),
                                proximity_weight=0.01, diversity_weight=0.01, device='cpu',
                                batch_norm_regularization=False):
    """
    idea from: http://www.bnikolic.co.uk/blog/pytorch/python/2021/02/28/pytorch-scipyminim.html

    Uses quasi-Newton optimization (Sequential Least Squares Programming) to find optimal input alteration for model
    according to:
    loss = predict_fn(y) + gamma * mean squared distance between optimized and original point (excluding fixed values)
           (optionally + delta * negative distance to points that should be avoided [Haldar2021])
    :param data_point:          numpy model input to optimize
    :param kept_feature_idx:    index of feature in data_point to keep, or None for not constraining any feature
                                Can also contain a list of indices to keep
    :param predict_fn:          function of pytorch model to optimize loss for
    :param proximity_weight:    float weight loss factor for proximity to the optimized input
    :param diversity_weight:    float weight loss factor for points_to_keep_distance
    :param points_to_keep_distance: list of numpy data points to keep distance from (distance added to loss function)
    :param batch_norm_regularization: bool whether to add regularization loss for batch norm layers
    :return:                    numpy optimized data_point
    """
    data_point = torch.autograd.Variable(torch.from_numpy(data_point.astype('float32')), requires_grad=True).to(device)
    proximity_weight = torch.Tensor([proximity_weight]).to(device)
    diversity_weight = torch.Tensor([diversity_weight]).to(device)
    if threshold is not None:
        threshold = torch.Tensor([threshold]).to(device)
    zero_elem = torch.tensor(0.0)
    if len(points_to_keep_distance) > 0:
        points_to_keep_distance = torch.tensor(np.concatenate([p.reshape([1, -1]) for p in points_to_keep_distance]),
                                               dtype=torch.float32, device=device)
    else:
        points_to_keep_distance = None

    def val_and_grad(x):

        pred_loss = predict_fn(x)
        if threshold is not None:  # hinge loss for anomaly score
            pred_loss = torch.max(zero_elem, pred_loss - threshold)
        prox_loss = proximity_weight * torch.linalg.vector_norm(data_point - x)
        if points_to_keep_distance is not None:
            divs_loss = diversity_weight * torch.max(-1 * torch.norm(points_to_keep_distance - x.repeat(len(points_to_keep_distance), 1), dim=1))
        else:
            divs_loss = 0
        loss = pred_loss + prox_loss + divs_loss

        loss.backward()
        grad = x.grad
        return loss, grad

    def func(x):
        """scipy needs flattened numpy array with float64, tensorflow tensors with float32"""
        return [vv.cpu().detach().numpy().astype(np.float64).flatten() for vv in
                val_and_grad(torch.tensor(x.reshape([1, -1]), dtype=torch.float32, requires_grad=True))]

    if kept_feature_idx is None:
        constraints = ()
    elif type(kept_feature_idx) == int:
        constraints = {'type': 'eq', 'fun': lambda x: x[kept_feature_idx] - data_point.detach().numpy()[:, kept_feature_idx]}
    elif len(kept_feature_idx) != 0:
        kept_feature_idx = np.where(kept_feature_idx)[0]
        constraints = []
        for kept_idx in kept_feature_idx:
            constraints.append(
                {'type': 'eq', 'fun': partial(lambda x, idx: x[idx] - data_point.detach().numpy()[:, idx], idx=kept_idx)})
    else:
        constraints = ()

    res = minimize(fun=func,
                   x0=data_point.detach().cpu(),
                   jac=True,
                   method='SLSQP',
                   constraints=constraints)
    opt_input = res.x.astype(np.float32).reshape([1, -1])

    return opt_input


def dynamic_synth_data(sample, maskMatrix, model, background_type):
    """
    Dynamically generate background "deletion" data for each synthetic data sample by minimizing model output.
    background_type == 'full':      Solves SLSQP for each mask, while constraining all features
                                    with maskMatrix == 1 to the original value.
    background_type == 'optimized': Finds most benign inputs through SLSQP while always constraining 1 feature,
                                    takes mean of all benign inputs when generating background for samples where
                                    more then 1 feature needs to be constrained (instead of solving SLSQP again).
    background_type == 'reliable':  Uses SLSQP for each mask, but also checks that each optimized sample is reliable
                                    (i.e. not an adversarial sample) by checking the hemisphere behind the
                                    optimized sample for anomalies. [Haldar2021]
    :param sample:                  np.ndarray sample to explain, shape (1, n_features)
    :param maskMatrix:              np.ndarray matrix with features to remove in SHAP sampling process
                                    1 := keep, 0 := optimize/remove
    :param model:                   ml-model to optimize loss for
    :param background_type:         String type of optimization, one of ['full', 'takeishi']
    :return:                        np.ndarray with synthetic samples, shape maskMatrix.shape

    Example1:
    dynamic_synth_data(sample=to_explain[0].reshape([1, -1]),
                    maskMatrix=maskMatrix,
                    model=load_model('../outputs/models/AE_cat'),
                    background_type='full')

    Example2:
    # integrate into SHAP in shap.explainers.kernel @ KernelExplainer.explain(), right before calling self.run()
    if self.dynamic_background:
        from xai.automated_background import dynamic_synth_data
        self.synth_data, self.fnull = dynamic_synth_data(sample=instance.x,
                                                        maskMatrix=self.maskMatrix,
                                                        model=self.full_model,
                                                        background_type=self.dynamic_background)
        self.expected_value = self.fnull
    """
    assert sample.shape[0] == 1, \
        f"Dynamic background implementation can't explain more then one sample at once, but input had shape {sample.shape}"
    assert maskMatrix.shape[1] == sample.shape[1], \
        f"Dynamic background implementation requires sampling of all features (omitted in SHAP when baseline[i] == sample[i]):\n" \
        f"shapes were maskMatrix: {maskMatrix.shape} and sample: {sample.shape}\n" \
        f"Use of np.inf vector as SHAP baseline is recommended"

    if background_type == 'full':
        # Optimize every data entry
        x_opt = np.empty(maskMatrix.shape)

        for idx, mask in tqdm.tqdm(enumerate(maskMatrix)):
            kept_feature_idxs = [i for i in range(mask.shape[0]) if mask[i] == 1]
            x_opt[idx] = optimize_input_quasi_newton(data_point=sample,
                                                     kept_feature_idx=kept_feature_idxs,
                                                     predict_fn=model)
        x_opt_no_mask = optimize_input_quasi_newton(data_point=sample,
                                                    kept_feature_idx=None,
                                                    predict_fn=model)
        fnull = model(x_opt_no_mask).unsqueeze(0).detach().numpy()
        return x_opt, fnull

    elif background_type in ['optimized']:
        # optimize all permutations with 1 kept variable, then aggregate results
        x_hat = []  # contains optimized feature (row) for each leave-one-out combo of varying features (column)
        # Sequential Least Squares Programming
        for kept_idx in tqdm.tqdm(range(sample.shape[1])):
            x_hat.append(optimize_input_quasi_newton(data_point=sample,
                                                     kept_feature_idx=kept_idx,
                                                     predict_fn=model,
                                                     batch_norm_regularization=False))
        x_hat.append(optimize_input_quasi_newton(data_point=sample,
                                                 kept_feature_idx=None,
                                                 predict_fn=model,
                                                 batch_norm_regularization=False))
        x_hat = np.concatenate(x_hat)

        # print("Debugging: ###############################################")
        # print("Old sample MSE:\t\t", MSE(y_true=sample, y_pred=model(sample)))
        # print("New sample MSE:\t\t", MSE(y_true=x_hat[-1].reshape([1, -1]), y_pred=model(x_hat[-1].reshape([1, -1]))))
        # print("Euclidean distance:\t", MSE(y_true=sample, y_pred=x_hat[-1].reshape([1, -1])))

        # Find x_tilde by adding x_hat entries for each feature to keep
        def sum_sample(row):
            S = x_hat[:-1][row == True]
            return ((S.sum(axis=0) + x_hat[-1]) / (S.shape[0] + 1)).reshape([1, -1])

        x_tilde_Sc = []
        for mask in maskMatrix:
            x_tilde_Sc.append(sum_sample(mask))
        x_tilde_Sc = np.concatenate(x_tilde_Sc)
        x_tilde = sample.repeat(maskMatrix.shape[0], axis=0) * maskMatrix + x_tilde_Sc * (1 - maskMatrix)

        fnull = model(torch.tensor(x_hat[-1]).unsqueeze(0)).detach().numpy()
        return x_tilde, fnull

    else:
        raise ValueError(
            f"Variable 'background_type' needs to be one of ['full', 'optimized', 'reliable'], but was {background_type}")
