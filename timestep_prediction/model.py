import torch as t
from helpers import augment, output_size
from shared_components import ClassificationHead, Body
import globals


class PairwiseMLP(t.nn.Module):
    """
    Function of two tensors
    """

    def __init__(self, input_size, projection_dim):
        super().__init__()
        self.layers = t.nn.Sequential(
            t.nn.Linear(input_size * 2, projection_dim),
            t.nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, ref, example):
        concat = t.cat([ref, example], dim=1)
        return self.layers(concat)


class SelfSupervisedHead(t.nn.Module):

    def __init__(self, input_dim=None, n_examples=3, projection_dim=globals.PROJECTION_DIM):
        super().__init__()
        if not input_dim:
            input_dim = output_size(n_timesteps=globals.WINDOW_SIZE,
                                    kernel_size=globals.CONV_SIZE,
                                    n_layers=len(globals.N_CHANNELS_CONV) - 1,
                                    pool_size=globals.POOL_SIZE,
                                    out_channels=globals.N_CHANNELS_CONV[-1])
        self.pairwise = PairwiseMLP(input_dim, projection_dim)
        self.layers = t.nn.Sequential(
            t.nn.Linear(projection_dim * n_examples, projection_dim),
            t.nn.ReLU(),
            t.nn.Linear(projection_dim, n_examples),
        )

    def forward(self, ref, examples):
        pairs = [self.pairwise(e, ref) for e in examples]
        concat = t.cat(pairs, dim=1)
        return self.layers(concat)


class HybridModel(t.nn.Module):

    def __init__(self,
                 random_augment=False,
                 embedding_dim=globals.EMBED_DIM,
                 n_examples=3):

        super().__init__()
        self.body = Body()
        self.head = SelfSupervisedHead(n_examples=n_examples)
        self.supervised_head = ClassificationHead(embed_dim=embedding_dim, use_hidden=True)
        self.n_examples = n_examples
        self.random_augment = random_augment

    def forward(self, ref, examples=None, is_supervised=True):
        if is_supervised and self.random_augment and self.training:
            with t.no_grad():
                ref = augment(ref)
        elif not is_supervised:
            assert len(examples) == self.n_examples, f"Must have {self.n_examples} for unsupervised surrogate task step"
            ref = self.body(ref)
            examples = [self.body(e) for e in examples]
            return self.head(ref, examples)
        x = self.body(ref)
        return self.supervised_head(x)
