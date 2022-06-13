import torch as t
from torchvision import transforms
from helpers import has_nans, get_data_from_csv, downsample_convert_to_tensor, RotationTransform, NoiseTransform, ChannelDeletionTransform, \
    write_and_print
from torch.utils.data import Dataset
import torch.nn.functional as F
from shared_components import Body, ProjectionHead
import einops
import globals


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with t.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ConvNetSimCLR(t.nn.Module):

    def __init__(self):
        super().__init__()
        self.body = Body()
        self.head = ProjectionHead(output_dim=globals.PROJECTION_DIM, hidden_dim=globals.SIMCLR_HIDDEN_DIM)

    def forward(self, x):
        return self.head(self.body(x))


class SimCLRDataset(Dataset):

    def __init__(self, files, mean, std, down_sample=globals.DOWN_SAMPLE, window_size=globals.WINDOW_SIZE, transform=None, step=globals.STEP):
        data = [get_data_from_csv(f) for f in files]
        data = [downsample_convert_to_tensor(f, down_sample) for f in data]
        data = [d for d in data if not has_nans(d)]
        data = t.cat(data, dim=0)
        x = data[:, :-1]
        x = (x - mean) / std
        self.data = x.float()
        self.window_size = window_size
        self.step = step
        self.transform = transform

    def __len__(self):
        return (self.data.shape[0] - self.window_size) // self.step

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.window_size
        window = self.data[start:end, :]
        window = einops.rearrange(window, 't c -> c t 1')
        if self.transform:
            window = self.transform(window)
        return window


class ContrastiveLearningViewGenerator(object):
    # Limit contrastive examples to same category of activity??
    """Copied from https://github.com/sthalles/SimCLR/"""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


class ContrastiveLearningDataset:
    """Adapted from https://github.com/sthalles/SimCLR/"""

    def __init__(self, files, mean, std, window_size=globals.WINDOW_SIZE, down_sample=globals.DOWN_SAMPLE):
        self.files = files
        self.mean = mean
        self.std = std
        self.window_size = window_size
        self.down_sample = down_sample

    @staticmethod
    def get_simclr_pipeline_transform(std):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        data_transforms = transforms.Compose([RotationTransform(std=std),
                                              ChannelDeletionTransform(channel_dim=1, n_channels=1),
                                              transforms.RandomErasing(p=1, scale=(0.1, 0.3)),
                                              NoiseTransform(scale=0.1)])
        return data_transforms

    def get_dataset(self, n_views):
        transform = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(std=30), n_views)
        return SimCLRDataset(self.files, self.mean, self.std, down_sample=self.down_sample, window_size=self.window_size, transform=transform)


class SimCLR(object):
    """
    Adapted from https://github.com/sthalles/SimCLR/
    """

    def __init__(self, model, optimizer, scheduler, device, batch_size, n_views, temperature, epochs, filename):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature
        self.filename = filename
        self.criterion = t.nn.CrossEntropyLoss().to(self.device)
        self.epochs = epochs

    def info_nce_loss(self, features):
        labels = t.cat([t.arange(self.batch_size) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        features = F.normalize(features, dim=1)
        similarity_matrix = t.matmul(features, features.T)
        # discard the main diagonal from both: labels and similarities matrix
        mask = t.eye(labels.shape[0], dtype=t.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = t.cat([positives, negatives], dim=1)
        labels = t.zeros(logits.shape[0], dtype=t.long).to(self.device)
        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader) -> ConvNetSimCLR:

        with open(self.filename, "w") as logfile:

            n_iter = 0
            write_and_print(f"Start SimCLR training for {self.epochs} epochs.", logfile)

            for epoch_counter in range(self.epochs):
                write_and_print(f"Epoch: {epoch_counter}", logfile)
                for batch in train_loader:
                    batch = t.cat(batch, dim=0)
                    batch = batch.to(self.device, dtype=t.float)
                    features = self.model(batch)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if n_iter % 100 == 0:
                        top1, top5 = accuracy(logits, labels, topk=(1, 5))
                        write_and_print(
                            f"n_iter: {n_iter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}\tTop5 accuracy: {top5[0]}\tLR: {self.scheduler.get_last_lr()[0]}",
                            logfile)
                    n_iter += 1
                # warmup for the first 3 epochs
                if epoch_counter >= 3:
                    self.scheduler.step()

            t.save(self.model.state_dict(), f"{self.filename.split('.')[0]}.pt")

            return self.model
