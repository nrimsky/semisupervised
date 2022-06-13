import torch as t
from torch import nn
from torch.nn import functional as F
import globals
from shared_components import Body


class VAE(t.nn.Module):

    def __init__(self, channels=globals.N_CHANNELS_CONV, latent_dim=globals.PROJECTION_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Body(channels=channels)
        flat_dim = self.encoder.flat_dim
        print(f"VAE has encoder output dimension: {flat_dim} and latent dimension: {latent_dim}")
        self.mean = nn.Linear(flat_dim, latent_dim)
        self.variance = nn.Linear(flat_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, flat_dim)
        reverse_channels = channels[::-1]
        kernel_size = self.encoder.conv[0].block[0].kernel_size
        self.decoder = nn.Sequential(
            # nn.ReLU(),
            nn.Unflatten(-1, (reverse_channels[0], flat_dim // channels[-1], 1)),
            *[nn.Sequential(
                t.nn.Upsample(scale_factor=(2, 1)),
                t.nn.ConvTranspose2d(reverse_channels[i], reverse_channels[i + 1], kernel_size),
                t.nn.ReLU()
            ) for i in range(len(reverse_channels) - 1)],
            nn.Sequential(
                t.nn.Upsample(size=(globals.WINDOW_SIZE - kernel_size[0] + 1, 1)),
                t.nn.ConvTranspose2d(reverse_channels[-1], reverse_channels[-1], kernel_size),
            )
        )

    def encode(self, input):
        """
        Encode the input and then transform into mean and std components of the latent Gaussian distribution
        """
        encoded = self.encoder(input)
        m = self.mean(encoded)
        v = self.variance(encoded)
        return [m, v]

    def decode(self, z):
        """
        Maps latent vector into the input space
        """
        decoder_input = self.decoder_input(z)
        decoded = self.decoder(decoder_input)
        return decoded

    @staticmethod
    def reparameterize(m, v):
        std = t.exp(0.5 * v)
        eps = t.randn_like(std)
        return eps * std + m

    def forward(self, input):
        m, v = self.encode(input)
        z = self.reparameterize(m, v)
        return [self.decode(z), input, m, v]

    @staticmethod
    def loss(recons, input, m, v, kld_weight=1) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """

        recons_loss = F.mse_loss(recons, input)
        kld_loss = -0.5 * (1 + v - (m ** 2) - v.exp())
        kld_loss = kld_weight * t.mean(t.sum(kld_loss, dim=1))
        loss = recons_loss + kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples):
        """
        Samples from the latent space and returns decoded version
        """
        z = t.randn(num_samples, self.latent_dim)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input, returns the reconstructed input
        """
        return self.forward(x)[0]
