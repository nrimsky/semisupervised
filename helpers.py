"""
Various helper methods and classes used in multiple training approaches
"""

import torch as t
import os
import pandas as pd
import einops
import numpy as np
import glob
from math import pi
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from io import TextIOWrapper
import globals
from math import floor


def get_swap_vec(left_feats, right_feats, swap_comps, half=True):
    swap_cols = [i for i in left_feats + right_feats for j in swap_comps if j in i]
    swap_inds = [(left_feats + right_feats).index(j) for j in swap_cols]
    swap_vec = np.ones(len(left_feats + right_feats))
    swap_vec[swap_inds] = -1

    if half:
        swap_vec = swap_vec[:len(left_feats)]

    return swap_vec


def get_mean_std(filenames: List[str]) -> Tuple[t.Tensor, t.Tensor]:
    """
    :param filenames: Filenames of CSV files with raw data
    :return: Mean and standard deviation to be used in data normalisation
    """
    data = []
    for f in filenames:
        df = pd.read_csv(f)
        df2 = df.copy()
        df2[[*globals.SWAP_COLS]] *= -1
        df = df[list(globals.X_COLS_LEFT) + list(globals.X_COLS_RIGHT)].values
        df2 = df2[list(globals.X_COLS_RIGHT) + list(globals.X_COLS_LEFT)].values
        data.append(df)
        data.append(df2)
    data = np.concatenate(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return t.tensor(mean), t.tensor(std)


def make_experiment_name(root: str, base_dir: str = "logs") -> str:
    """
    :param root: A descriptive name for the experiment
    :param base_dir: Directory where to save logs
    :return: Filename of txt file where logs can be printed
    """
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    return os.path.join(base_dir, f"{root}_{dt_string}.txt")


def degrees2rads(x: float) -> float:
    """
    :param x: Angle in degrees
    :return: Angle in radians
    """
    return pi * x / 180


def general_rotation_matrix(alpha, beta, gamma) -> t.Tensor:
    """
    :param alpha: Random angle
    :param beta: Random angle
    :param gamma: Random angle
    :return: Random rotation matrix that can be applied to one leg's data
    """
    r = t.tensor([[t.cos(beta) * t.cos(gamma),
                   t.sin(alpha) * t.sin(beta) * t.cos(gamma) - t.cos(alpha) * t.sin(gamma),
                   t.sin(alpha) * t.sin(gamma) + t.cos(alpha) * t.sin(beta) * t.cos(gamma)],
                  [t.cos(beta) * t.sin(gamma),
                   t.cos(alpha) * t.cos(gamma) + t.sin(alpha) * t.sin(beta) * t.sin(gamma),
                   - t.sin(alpha) * t.cos(gamma) + t.cos(alpha) * t.sin(beta) * t.sin(gamma)],
                  [-t.sin(beta), t.sin(alpha) * t.cos(beta), t.cos(alpha) * t.cos(beta)]])
    return r


def augment(x: t.Tensor, std=10) -> t.Tensor:
    """
    :param x: Original data tensor (optional_batch) x n_sensors x n_timesteps x ...
    :param std: Standard deviation of rotation angle in degrees
    :return: Randomly rotated tensor
    """

    def batch_mul(t1, r):
        return t.matmul(r, t1.transpose(-2, -3)).transpose(-2, -3)

    def mul(t1, r):
        return t.matmul(r, t1.transpose(-2, -3)).transpose(-2, -3)

    if x.shape[1] == 12:
        dim = 1
    elif x.shape[0] == 12:
        dim = 0
    else:
        raise
    device = "cuda" if x.is_cuda else "cpu"
    alphas = t.normal(mean=0, std=std, size=(2,))
    betas = t.normal(mean=0, std=std, size=(2,))
    gammas = t.normal(mean=0, std=std, size=(2,))
    r1 = general_rotation_matrix(degrees2rads(alphas[0]), degrees2rads(betas[0]), degrees2rads(gammas[0])).to(device)
    r2 = general_rotation_matrix(degrees2rads(alphas[1]), degrees2rads(betas[1]), degrees2rads(gammas[1])).to(device)
    if dim == 0:
        s1 = t.cat([mul(x[:3, ...], r1), mul(x[3:6, ...], r1)], dim=dim)
        s2 = t.cat([mul(x[6:9, ...], r2), mul(x[9:12, ...], r2)], dim=dim)
    else:
        s1 = t.cat([batch_mul(x[:, :3, ...], r1), batch_mul(x[:, 3:6, ...], r1)], dim=dim)
        s2 = t.cat([batch_mul(x[:, 6:9, ...], r2), batch_mul(x[:, 9:12, ...], r2)], dim=dim)
    return t.cat([s1, s2], dim=dim)


def get_files_for_subjects(subjects: List[int], base_dir="data") -> List[str]:
    """
    :param subjects: Subject ids (range 1  - 25)
    :return: Filenames for those subject ids (raw csv data files)
    """
    files = glob.glob(os.path.join(base_dir, "*.csv"))
    returned = []
    for f in files:
        if int(os.path.split(f)[-1].split("_")[0]) in subjects:
            returned.append(f)
    return returned


def ensure_dir(directory: str, empty=True):
    """
    :param directory: Directory name we need to exist
    :param empty: Whether to empty the directory if it has things in it
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist - making {directory}")
        os.makedirs(directory)
    elif len(os.listdir(directory)) != 0 and empty:
        print(f"Directory {directory} is not empty - emptying {directory}")
        for f in os.listdir(directory):
            os.remove(os.path.join(directory, f))


def transform_right(df1: pd.DataFrame, swap_cols: List[str]):
    """
    :param df1: Original dataframe
    :param swap_cols: Columns of dataframe to be reflected
    :return: Dataframe with right leg transformed to match sensor orientation of left leg
    """
    df = df1.copy()
    for col in swap_cols:
        df[col] *= -1
    return df


def get_tensors(directory: str, return_filenames=False):
    """
    :param directory: Directory where preprocessed tensors of data have been saved
    :param return_filenames: Whether to return filenames in that directory
    :return: Loaded tensors
    """
    print(f"Loading tensors from files in {directory}")
    d = directory
    filenames = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and "mean" not in f and "std" not in f]
    tensors = [t.load(f) for f in filenames]
    if return_filenames:
        return tensors, filenames
    return tensors


def format_loss(loss: float) -> float:
    """
    :param loss: loss
    :return: nicely formatted loss
    """
    return round(loss * 10000) / 10000


def get_data_from_csv(filename,
                      swap_cols=globals.SWAP_COLS,
                      x_cols_right=globals.X_COLS_RIGHT,
                      x_cols_left=globals.X_COLS_LEFT,
                      y_col_right=globals.Y_COL_RIGHT,
                      y_col_left=globals.Y_COL_LEFT) -> np.array:
    """
    :param filename: Filename of CSV file with raw sensor data
    :param swap_cols: Columns to reflect to make right leg sensor orientation match left leg
    :param x_cols_right: Data columns for right leg sensors
    :param x_cols_left: Data columns for left leg sensors
    :param y_col_right: Label column for right leg gait cycle stage
    :param y_col_left: Label column for left leg gait cycle stage
    :return: numpy array with data in a single coordinate frame with primary leg as label
    """
    data = pd.read_csv(filename)
    columns = list(x_cols_right) + list(x_cols_left) + [y_col_right]
    right_primary_x = data[columns].copy()
    right_primary_x = transform_right(right_primary_x, swap_cols).values
    columns = list(x_cols_left) + list(x_cols_right) + [y_col_left]
    left_primary_x = data[columns].values
    all_data = np.concatenate([right_primary_x, left_primary_x], axis=0)
    return all_data


def get_data_from_csv_with_metadata(filename,
                                    swap_cols=globals.SWAP_COLS,
                                    x_cols_right=globals.X_COLS_RIGHT,
                                    x_cols_left=globals.X_COLS_LEFT,
                                    y_col_right=globals.Y_COL_RIGHT,
                                    y_col_left=globals.Y_COL_LEFT) -> np.array:
    """
    :param filename: Filename of CSV file with raw sensor data
    :param swap_cols: Columns to reflect to make right leg sensor orientation match left leg
    :param x_cols_right: Data columns for right leg sensors
    :param x_cols_left: Data columns for left leg sensors
    :param y_col_right: Label column for right leg gait cycle stage
    :param y_col_left: Label column for left leg gait cycle stage
    :return: numpy array with data in a single coordinate frame with primary leg as label
    """
    main_data = get_data_from_csv(filename, swap_cols, x_cols_right, x_cols_left, y_col_right, y_col_left)
    csv_name = os.path.split(filename)[-1]
    name_parts = csv_name.split("_")
    subject = name_parts[0]
    activity = name_parts[1]
    return main_data, subject, activity


def downsample_convert_to_tensor(data: np.array, down_sample: int) -> t.Tensor:
    """
    :param data: Numpy array with preprocessed sensor data
    :param down_sample: Downsampling factor
    :return: Downsampled data as torch tensor
    """
    data = t.tensor(data, dtype=t.float64)
    new_l = (data.shape[0] // down_sample) * down_sample
    new_idx = t.arange(0, new_l, down_sample)
    data = data[new_idx]
    return data


def has_nans(tensor: t.Tensor) -> bool:
    """
    :param tensor: Torch tensor
    :return: true if tensor contains nans
    """
    return t.isnan(tensor).sum().item() > 0


def cyclical_err(pred, labels):
    """
    :param pred:
    :param labels: True labels
    :return: Error measured using cyclical formula
    """
    d1 = t.abs(pred - labels)
    d2 = 12 - t.abs((pred - labels))
    min, _ = t.min(t.stack([d1, d2]), dim=0)
    min[pred == 12] = 3
    min[labels == 12] = 3
    return t.mean(min.float())


def evaluate(model: t.nn.Module, data_loader: DataLoader, max_batches: int = None, use_circ=False):
    """
    :param model: Classification model
    :param data_loader: Dataloader that returns data window, label pairs
    :param max_batches: Maximum number of batches to use during evaluation
    :param use_circ: Whether to also return the circular accuracy metric
    :return: % Accuracy or % Accuracy, Avg circular distance
    """
    acc = 0
    c_acc = 0
    n = 0
    with t.no_grad():
        for idx, (batch, labels) in enumerate(data_loader):
            if max_batches:
                if idx >= max_batches:
                    break
            batch = batch.to("cuda", dtype=t.float)
            labels = labels.to("cuda", dtype=t.long)
            out = model(batch)
            if type(out) == tuple:
                out = out[0]
            pred = t.argmax(out, dim=-1)
            batch_acc = t.sum(t.eq(pred, labels)) / labels.shape[0]
            if use_circ:
                c_acc += cyclical_err(pred=pred, labels=labels)
            acc += batch_acc
            n += 1
    _acc = (acc / t.tensor(n)).item()
    _c_acc = (c_acc / t.tensor(n)).item()
    if use_circ:
        return _acc, _c_acc
    return _acc


def get_loaders(unlabelled_subjects: List[int], labelled_subjects: List[int], base_dir="../data", batch_size=64, test_size=0.5) -> Tuple[
    DataLoader, DataLoader, DataLoader]:
    """
    :param unlabelled_subjects: Subjects to be used as unlabelled data (target domain in the case of DA)
    :param labelled_subjects: Subjects to be used as labelled data (source domain in the case of DA)
    :param base_dir: Directory where raw data located
    :param batch_size: Batch size
    :param test_size: proportion of unlabelled data used for testing
    :return: Dataloaders for labelled and unlabelled subjects
    """
    unlabelled_files = get_files_for_subjects(unlabelled_subjects, base_dir=base_dir)
    labelled_files = get_files_for_subjects(labelled_subjects, base_dir=base_dir)
    mean, std = get_mean_std(labelled_files)
    unlabelled_dataset = ClassificationDataset(unlabelled_files, mean, std)
    labelled_dataset = ClassificationDataset(labelled_files, mean, std)
    unlabelled_train_idx, test_idx = train_test_split(list(range(len(unlabelled_dataset))), test_size=test_size)
    unlabelled_loader = DataLoader(Subset(unlabelled_dataset, unlabelled_train_idx), batch_size=batch_size, shuffle=True, pin_memory=True,
                                   drop_last=True)
    test_loader = DataLoader(Subset(unlabelled_dataset, test_idx), batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
    labelled_loader = DataLoader(labelled_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return unlabelled_loader, labelled_loader, test_loader


def get_loaders_just_tt(train_subjects: List[int], test_subjects: List[int], base_dir="../data", batch_size=64) -> Tuple[DataLoader, DataLoader]:
    """
    :param train_subjects: Subjects to be used as training data
    :param test_subjects: Subjects to be used as test data
    :param base_dir: Directory where raw data located
    :param batch_size: Batch size
    :return: Dataloaders for labelled and unlabelled subjects
    """
    train_files = get_files_for_subjects(train_subjects, base_dir=base_dir)
    test_files = get_files_for_subjects(test_subjects, base_dir=base_dir)
    mean, std = get_mean_std(train_files)
    train_dataset = ClassificationDataset(train_files, mean, std)
    test_dataset = ClassificationDataset(test_files, mean, std)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
    return train_loader, test_loader


def write_and_print(text: str, file: TextIOWrapper):
    """
    :param text: Text to print to the console and write to the file
    :param file: File handle to write to
    :return:
    """
    file.write(f"{text}\n")
    print(text)


class ClassificationDataset(Dataset):
    """
    Pytorch dataset with time windows of tensor data with gait cycle stage labels that correspond to the end of that time window
    _______L
      _______L
        _______L
    | | <- step
    |      | <- window_size
    L = label location
    """

    def __init__(self, files, mean, std, down_sample=globals.DOWN_SAMPLE, window_size=globals.WINDOW_SIZE, step=globals.STEP):
        data = [get_data_from_csv(f) for f in files]
        data = [downsample_convert_to_tensor(f, down_sample) for f in data]
        data = [d for d in data if not has_nans(d)]
        data = t.cat(data, dim=0)
        x = data[:, :-1]
        y = data[:, -1]
        x = (x - mean) / std
        self.data = x.float()
        self.labels = y.long()
        self.window_size = window_size
        self.step = step

    def __len__(self):
        return (len(self.labels) - self.window_size) // self.step

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.window_size
        window, label = self.data[start:end, :], self.labels[end]
        window = einops.rearrange(window, 't c -> c t 1')
        return window, label


class RotationTransform(object):
    """
    Composable transform that randomly rotates tensor
    """

    def __init__(self, std=globals.AUGMENT_STD):
        super().__init__()
        self.std = std

    def __call__(self, sample):
        return augment(sample, self.std)


class NoiseTransform(object):
    """
    Composable transform that adds random noise
    Provided data should be normalised (so has mean 0 and variance 1)
    """

    def __init__(self, scale=0.8):
        super().__init__()
        self.scale = scale

    def __call__(self, sample):
        return sample + t.randn_like(sample) * self.scale


class ChannelDeletionTransform(object):
    """
    Randomly deletes part / all of a channel
    """

    def __init__(self, channel_dim=1, n_channels=1):
        """
        :param channel_dim: Which dimension is the channel dimension
        :param n_channels: How many channels to delete
        """
        super().__init__()
        self.n_channels = n_channels
        self.channel_dim = channel_dim
        assert channel_dim < 3

    def __call__(self, sample):
        tot_n_channels = sample.shape[self.channel_dim]
        channels = t.randperm(tot_n_channels)[:self.n_channels]
        if self.channel_dim == 0:
            sample[channels] = 0
        elif self.channel_dim == 1:
            sample[:, channels] = 0
        elif self.channel_dim == 2:
            sample[:, :, channels] = 0
        return sample


def output_size(n_timesteps, kernel_size, n_layers, pool_size, out_channels, use_pool=True):
    h_in = n_timesteps
    w_in = 1

    def apply_conv(h, w):
        h_o = h - kernel_size[0] + 1
        w_o = w - kernel_size[1] + 1
        return h_o, w_o

    def apply_pool(h, w):
        h_o = floor(h / pool_size[0])
        w_o = floor(w / pool_size[1])
        return h_o, w_o

    h_out = h_in
    w_out = w_in

    for _ in range(n_layers):
        h_out, w_out = apply_conv(h_out, w_out)
        if use_pool:
            h_out, w_out = apply_pool(h_out, w_out)

    return out_channels * h_out * w_out


class SubjectActivityAwareDataset(Dataset):

    def __init__(self, files, mean, std, down_sample=globals.DOWN_SAMPLE, window_size=globals.WINDOW_SIZE, step=globals.STEP, relabel_dict=None):
        _data = [get_data_from_csv_with_metadata(f) for f in files]
        data = [d[0] for d in _data]
        subject = [int(d[1]) for d in _data]
        if relabel_dict:
            subject = [relabel_dict[s] for s in subject]
        activity = [d[2] for d in _data]
        act_set = set(activity)
        self.n_subjects = len(set(subject))
        self.n_activity = len(act_set)
        act_map = {a: i for i, a in enumerate(act_set)}  # map each activity to a unique int
        data = [downsample_convert_to_tensor(f, down_sample) for f in data]
        filt_data = []
        filt_subj = []
        filt_act = []
        for i, d in enumerate(data):
            if not has_nans(d):
                filt_data.append(d)
                filt_subj.append(subject[i] * t.ones(len(d)))  # int
                filt_act.append(act_map[activity[i]] * t.ones(len(d)))  # int
        data = t.cat(filt_data, dim=0)
        x = data[:, :-1]
        y = data[:, -1]
        x = (x - mean) / std
        self.data = x.float()
        self.labels = y.long()
        self.window_size = window_size
        self.step = step
        self.subjects = t.cat(filt_subj, dim=0).long()
        self.activities = t.cat(filt_act, dim=0).long()

    def __len__(self):
        return (len(self.labels) - self.window_size) // self.step

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.window_size
        window, label, subject, activity = self.data[start:end, :], self.labels[end], self.subjects[end], self.activities[end]
        window = einops.rearrange(window, 't c -> c t 1')
        return window, label, subject, activity


class BaseTrainer:

    def train(self, unsupervised_subjects: List[int], supervised_subjects: List[int], filename: str) -> float:
        return -1.0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
