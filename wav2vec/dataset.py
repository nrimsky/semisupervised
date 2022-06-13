import torch as t
from torch.utils.data import Dataset, DataLoader
from helpers import get_data_from_csv, downsample_convert_to_tensor, has_nans, get_files_for_subjects, get_mean_std
from globals import DOWN_SAMPLE
from typing import List, Tuple
from random import shuffle


class TimeWindowDataset(Dataset):

    def __init__(self, files, mean, std, n_timesteps, down_sample=DOWN_SAMPLE, step=10):
        data = [get_data_from_csv(f) for f in files]
        data = [downsample_convert_to_tensor(f, down_sample) for f in data]
        data = [d for d in data if not has_nans(d)]
        data = t.cat(data, dim=0)
        x = data[:, :-1]
        y = data[:, -1]
        x = (x - mean) / std
        self.data = x.float()
        self.labels = y.long()
        self.n_timesteps = n_timesteps
        self.step = step

    def __len__(self):
        return ((len(self.labels) - self.n_timesteps)//self.step)-1

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.n_timesteps
        window, label = self.data[start:end, :], self.labels[end]
        return window, label


def get_loaders(unlabelled: List[int], labelled: List[int], n_timesteps_super, n_timesteps_unsuper, base_dir="../data", step=10) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    :param unlabelled: Unlabelled subjects
    :param labelled: Labelled subjects
    :param n_timesteps_unsuper: Number of timesteps in unsupervised window
    :param n_timesteps_super: Number of timesteps in supervised window
    :param base_dir: Directory where raw data located
    :param step: Gap between time windows
    :return: Dataloaders for train and test subjects
    """
    unlabelled_files = get_files_for_subjects(unlabelled, base_dir=base_dir)
    l = len(unlabelled_files)
    shuffle(unlabelled_files)
    labelled_files = get_files_for_subjects(labelled, base_dir=base_dir)
    mean, std = get_mean_std(labelled_files)
    unsuper_dataset = TimeWindowDataset(unlabelled_files[:l//2], mean, std, n_timesteps=n_timesteps_unsuper, step=step)
    super_dataset = TimeWindowDataset(labelled_files, mean, std, n_timesteps=n_timesteps_super, step=step)
    test_dataset = TimeWindowDataset(unlabelled_files[l//2:], mean, std, n_timesteps=n_timesteps_super, step=step)
    unsuper_train_loader = DataLoader(unsuper_dataset, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
    super_train_loader = DataLoader(super_dataset, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=True)
    return unsuper_train_loader, super_train_loader, test_loader
