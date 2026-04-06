import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_string(log, string):
    print(string, flush=True)
    if log is not None:
        log.write(string + '\n')
        log.flush()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def symmetric_normalize(adj: torch.Tensor) -> torch.Tensor:
    rowsum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(rowsum.clamp(min=1e-12), -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


def get_adjacency_matrix(distance_df_filename, normalize=True, threshold=0.1):
    df = pd.read_csv(distance_df_filename, header=None)
    dist_mx = df.values.astype(np.float32)

    sigma = np.std(dist_mx)
    if sigma < 1e-6:
        sigma = 1.0

    adj_mx = np.exp(-np.square(dist_mx) / (2 * sigma ** 2))
    adj_mx[adj_mx < threshold] = 0.0
    np.fill_diagonal(adj_mx, 1.0)

    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32)

    if normalize:
        adj_tensor = symmetric_normalize(adj_tensor)

    return adj_tensor


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=False):
        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            if num_padding > 0:
                x_padding = np.repeat(xs[-1:], num_padding, axis=0)
                y_padding = np.repeat(ys[-1:], num_padding, axis=0)
                xs = np.concatenate([xs, x_padding], axis=0)
                ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean, std, fill_zeros=False):
        self.mean = mean
        self.std = std if std > 1e-8 else 1.0
        self.fill_zeros = fill_zeros

    def transform(self, data):
        data = data.copy()
        if self.fill_zeros:
            mask = (data == 0)
            data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir, batch_size, valid_batch_size, test_batch_size, fill_zeros=False):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    scaler = StandardScaler(
        mean=data['x_train'][..., 0].mean(),
        std=data['x_train'][..., 0].std(),
        fill_zeros=fill_zeros
    )

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler

    return data


def _get_mask(labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask = mask / (torch.mean(mask) + 1e-8)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_mse(preds, labels, null_val=np.nan):
    mask = _get_mask(labels, null_val)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val) + 1e-12)


def masked_mae(preds, labels, null_val=np.nan):
    mask = _get_mask(labels, null_val)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_maape(preds, labels, null_val=np.nan):
    mask = _get_mask(labels, null_val)
    loss = torch.arctan(torch.abs((preds - labels) / (torch.abs(labels) + 1e-5)))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_huber(preds, labels, delta=1.0, null_val=np.nan):
    mask = _get_mask(labels, null_val)

    error = preds - labels
    abs_error = torch.abs(error)

    delta_tensor = torch.tensor(delta, dtype=preds.dtype, device=preds.device)
    quadratic = torch.minimum(abs_error, delta_tensor)
    linear = abs_error - quadratic

    loss = 0.5 * quadratic ** 2 + delta_tensor * linear
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_r2(preds, labels, null_val=np.nan):
    mask = _get_mask(labels, null_val)

    preds = preds * mask
    labels = labels * mask

    mean_labels = torch.sum(labels) / (torch.sum(mask) + 1e-8)

    ss_res = torch.sum((labels - preds) ** 2)
    ss_tot = torch.sum((labels - mean_labels) ** 2)

    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2


def metric(pred, real):
    import numpy as np
    import torch

    # 兼容 numpy
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(real, np.ndarray):
        real = torch.from_numpy(real).float()

    mae = masked_mae(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()

    return mae, rmse