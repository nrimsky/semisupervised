import torch as t
import matplotlib.pyplot as plt
from helpers import augment, output_size
from math import pi
import globals
from torch.nn.functional import normalize


class ConvBodyBlock(t.nn.Module):

    def __init__(self, in_channels, out_channels, filter_shape=globals.CONV_SIZE, avg_pool_shape=globals.POOL_SIZE):
        super().__init__()
        self.block = t.nn.Sequential(
            t.nn.Conv2d(in_channels, out_channels, filter_shape),
            t.nn.ReLU(),
            t.nn.AvgPool2d(avg_pool_shape),
        )

    def forward(self, x):
        return self.block(x)


class Body(t.nn.Module):

    def __init__(self, channels=globals.N_CHANNELS_CONV, filter_shape=globals.CONV_SIZE, avg_pool_shape=globals.POOL_SIZE):
        super().__init__()
        self.conv = t.nn.Sequential(
            *[ConvBodyBlock(channels[i - 1], channels[i], filter_shape=filter_shape, avg_pool_shape=avg_pool_shape) for i in range(1, len(channels))],
        )
        self.flatten = t.nn.Flatten()
        self.flat_dim = output_size(n_timesteps=globals.WINDOW_SIZE,
                                    kernel_size=filter_shape,
                                    n_layers=len(channels) - 1,
                                    pool_size=avg_pool_shape,
                                    out_channels=channels[-1])

    def forward(self, x):
        x = self.conv(x)
        return self.flatten(x)


class ProjectionHead(t.nn.Module):
    def __init__(self, input_dim=None, output_dim=globals.PROJECTION_DIM, hidden_dim=None):
        super().__init__()
        if not input_dim:
            input_dim = output_size(n_timesteps=globals.WINDOW_SIZE,
                                    kernel_size=globals.CONV_SIZE,
                                    n_layers=len(globals.N_CHANNELS_CONV) - 1,
                                    pool_size=globals.POOL_SIZE,
                                    out_channels=globals.N_CHANNELS_CONV[-1])
        if hidden_dim:
            self.linear = t.nn.Sequential(
                t.nn.Linear(input_dim, hidden_dim),
                t.nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.linear = t.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class ClassificationHead(t.nn.Module):
    def __init__(self, input_dim=None, embed_dim=globals.EMBED_DIM, num_classes=13, use_dropout=False, use_hidden=True, hidden_dim=globals.PROJECTION_DIM):
        super().__init__()
        if not input_dim:
            input_dim = output_size(n_timesteps=globals.WINDOW_SIZE,
                                    kernel_size=globals.CONV_SIZE,
                                    n_layers=len(globals.N_CHANNELS_CONV) - 1,
                                    pool_size=globals.POOL_SIZE,
                                    out_channels=globals.N_CHANNELS_CONV[-1])
        self.dropout = t.nn.Dropout(0.1)
        if use_hidden:
            self.linear = t.nn.Sequential(
                t.nn.Linear(input_dim, hidden_dim),
                t.nn.ReLU(),
                t.nn.Linear(hidden_dim, embed_dim)
            )
        else:
            self.linear = t.nn.Linear(input_dim, embed_dim)
        self.embedding = t.nn.Parameter(t.rand(num_classes, embed_dim))
        self.cos_sim = t.nn.CosineSimilarity(dim=-1)
        self.use_dropout = use_dropout

    @staticmethod
    def circle_init(n_labels, embed_dim):
        angle = t.tensor((2 * pi) / (n_labels - 1))
        circle_inits = t.cat((t.tensor([[t.sin(angle * i), t.cos(angle * i)] for i in range(n_labels - 1)]), t.tensor([[0.001, 0.001]]))) + t.tensor(
            [[0.001, 0.001]])
        base = t.rand(n_labels, embed_dim - 2) * 0.05
        return t.cat((circle_inits, base), dim=1)

    def forward(self, x):
        if self.use_dropout and self.training:
            x = self.dropout(x)
        embedded = self.linear(x)
        return self.cos_sim(embedded.unsqueeze(-2), self.embedding)

    def visualise_embeddings(self, filename="embeddings"):
        """
        Projects embeddings down to 2D using PCA and saves visualisation to png file
        Uses rainbow colours to represent gait cycle stages (red -> purple for 1 -> 13)
        :param filename: File name to save figure
        """
        with t.no_grad():
            embedding = normalize(self.embedding.cpu().detach(), dim=-1)
            projected = t.matmul(embedding, t.pca_lowrank(embedding)[-1])
            plt.figure(figsize=(8, 8))
            for i in range(projected.shape[0]):
                plt.text(projected[i][0], projected[i][1], f"{i + 1}", color='black', size='x-large')
                plt.plot(projected[i][0], projected[i][1], color=globals.RAINBOW[i], marker='o', markersize=10)
            plt.savefig(f'{filename}.png')


class Model(t.nn.Module):

    def __init__(self, use_dropout=False, augment_input=False, augment_std=globals.AUGMENT_STD, use_hidden=True, channels=globals.N_CHANNELS_CONV, filter_shape=globals.CONV_SIZE, avg_pool_shape=globals.POOL_SIZE, head_hidden_dim=globals.PROJECTION_DIM):
        super().__init__()
        self.augment = augment_input
        self.body = Body(channels=channels, filter_shape=filter_shape, avg_pool_shape=avg_pool_shape)
        flat_dim = self.body.flat_dim
        self.head = ClassificationHead(input_dim=flat_dim, use_dropout=use_dropout, use_hidden=use_hidden, hidden_dim=head_hidden_dim)
        self.augment_std = augment_std

    def forward(self, x):
        if self.augment and self.training:
            with t.no_grad():
                augmented = augment(x, self.augment_std)
            return self.head(self.body(augmented))
        return self.head(self.body(x))
