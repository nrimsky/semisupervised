import torch as t
from shared_components import ClassificationHead
import einops


class Encoder(t.nn.Module):
    """
    Encodes n_timesteps time steps
    """

    def __init__(self, n_timesteps, output_size):
        super().__init__()
        self.layers = t.nn.Sequential(
            t.nn.Flatten(),
            t.nn.Linear(12 * n_timesteps, 16),
            t.nn.ReLU(),
            t.nn.Linear(16, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class Context(t.nn.Module):
    """
    Contextualises multiple encoded time windows
    """
    def __init__(self, n_sections, section_size, output_size):
        super().__init__()
        self.linear = t.nn.Sequential(
            t.nn.Flatten(),
            t.nn.Linear(n_sections * section_size, output_size),
            t.nn.ReLU()
        )

    def forward(self, sections):
        return self.linear(sections)


class StepSpecificTransform(t.nn.Module):
    """
    Step-k specific transformation of the context vector
    """
    def __init__(self, context_size, output_size):
        super().__init__()
        self.linear = t.nn.Sequential(t.nn.Linear(context_size, output_size),
                                      t.nn.ReLU(),
                                      t.nn.Linear(output_size, output_size))

    def forward(self, context):
        return self.linear(context)


class SelfSupervisedModel(t.nn.Module):

    def __init__(self, n_timesteps_z, n_sections_context, context_output_size, encoder_output_size, n_step_specific):
        super().__init__()
        self.step_specific = t.nn.Sequential(*[StepSpecificTransform(context_size=context_output_size, output_size=encoder_output_size) for _ in range(n_step_specific)])
        self.encoder = Encoder(n_timesteps=n_timesteps_z, output_size=encoder_output_size)
        self.context = Context(n_sections=n_sections_context, section_size=encoder_output_size, output_size=context_output_size)
        self.n_timesteps_z = n_timesteps_z
        self.n_sections_context = n_sections_context
        self.cos_sim = t.nn.CosineSimilarity(dim=-1)
        self.n_step_specific = n_step_specific
        self.context_output_size = context_output_size

    def forward(self, x):
        sections = einops.rearrange(x, 'x (a b) c -> (x a) b c', b=self.n_timesteps_z)
        encoded = self.encoder(sections)
        encoded = einops.rearrange(encoded, '(b t) enc_dim -> b t enc_dim', b=x.shape[0])
        context = self.context(encoded[:, :self.n_sections_context, :])
        all_scores = []
        for step in range(self.n_step_specific):
            transformed_context = self.step_specific[step](context)
            location = self.n_sections_context + step
            sampled = encoded[:, t.randperm(self.n_step_specific), :]
            sampled[:, step, :] = encoded[:, location, :]  # Insert positive sample in correct place
            scores = self.cos_sim(sampled, transformed_context.unsqueeze(-2))
            all_scores.append(scores)
        return t.cat(all_scores, dim=0)


class SupervisedModel(t.nn.Module):

    def __init__(self, self_supervised_model: SelfSupervisedModel):
        super().__init__()
        self.encoder = self_supervised_model.encoder
        self.context = self_supervised_model.context
        self.n_timesteps_z = self_supervised_model.n_timesteps_z
        self.head = ClassificationHead(input_dim=self_supervised_model.context_output_size, use_hidden=False, use_dropout=False)

    def forward(self, x):
        sections = einops.rearrange(x, 'x (a b) c -> (x a) b c', b=self.n_timesteps_z)
        encoded = self.encoder(sections)
        encoded = einops.rearrange(encoded, '(b t) enc_dim -> b t enc_dim', b=x.shape[0])
        context = self.context(encoded)
        return self.head(context)