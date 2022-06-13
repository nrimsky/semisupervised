import torch as t
from torch.utils.data import Dataset
from helpers import get_data_from_csv, downsample_convert_to_tensor, has_nans
import numpy as np
import einops
import globals


class TimestepPredictionDataset(Dataset):

    def __init__(self, files, mean, std, shift_pos, shift_neg_std, down_sample=globals.DOWN_SAMPLE, window_size=globals.WINDOW_SIZE, step=globals.STEP, n_samples=3):
        """
        :param files: Filenames where raw csv data located
        :param mean: Mean for normalising (from labelled subjects)
        :param std: Standard deviation for normalising (from labelled subjects)
        :param shift_pos: Positive shift - self supervised training task is tying to ascertain which sample was shifted by this amount
        :param shift_neg_std: Standard deviation of negative shifts (centered around shift_pos)
        :param down_sample: Downsampling factor
        :param window_size: Number of timesteps passed to model per label
        :param step: Number of samples between consecutive reference samples
        :param n_samples: How many samples the model has to choose between
        """
        data = [self.process_csv(f, down_sample, mean, std) for f in files]
        self.X = t.cat([d for d in data if d is not None], dim=0)
        self.window_size = window_size
        self.step = step
        self.shift_pos = shift_pos
        self.shift_neg_std = shift_neg_std
        self.n_samples = n_samples

    @staticmethod
    def process_csv(filename, downsample, mean, std):
        data = get_data_from_csv(filename)
        data = downsample_convert_to_tensor(data, downsample)
        data = data[..., :-1]
        data -= mean
        data /= std
        if not has_nans(data):
            return data
        return None

    def __len__(self):
        return (len(self.X) - self.window_size - self.shift_pos - self.step) // self.step

    def __getitem__(self, idx):
        start_idx = idx * self.step
        ref_window = self.X[start_idx: start_idx + self.window_size]
        pos_window = self.X[start_idx + self.shift_pos: start_idx + self.window_size + self.shift_pos]
        neg_shifts = []
        for _ in range(self.n_samples - 1):
            shift = int(np.random.normal(self.shift_pos, self.shift_neg_std))
            if start_idx + shift + self.window_size > len(self.X) - 1 or start_idx + shift < 0:
                # Some sensible default if index out of range - should happen only ~ once per dataset
                neg_shifts.append(self.step)
            else:
                neg_shifts.append(shift)
        neg_windows = []
        for neg_shift in neg_shifts:
            neg_windows.append(self.X[start_idx + neg_shift: start_idx + self.window_size + neg_shift])
        all_windows = t.stack([ref_window, pos_window] + neg_windows)
        permutation = t.randperm(self.n_samples) + 1
        label = (permutation == 1).nonzero()[0][0]
        all_windows[1:] = all_windows[permutation]
        all_windows = einops.rearrange(all_windows, 'w t s -> w s t 1')
        return all_windows.float(), label