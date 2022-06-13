import torch as t
from model import VAE
from helpers import get_loaders_just_tt, format_loss


class Trainer:

    def __init__(self, model: VAE, kld_weight=1.0, lr=0.001):
        self.model = model
        self.kld_weight = kld_weight
        self.lr = lr

    def forward(self, inp):
        return self.model(inp)

    def train_step(self, batch):
        inp, labels = batch
        results = self.forward(inp.cuda())
        train_loss = self.model.loss(*results, kld_weight=self.kld_weight)
        return train_loss['loss'], train_loss['Reconstruction_Loss'], train_loss['KLD']

    def validation_step(self, batch):
        inp, labels = batch
        results = self.forward(inp.cuda())
        val_loss = self.model.loss(*results, kld_weight=self.kld_weight)
        print("Validation: "+" ".join(f"{key} : {val.item()}" for key, val in val_loss.items()))

    def sample(self, n):
        return self.model.sample(n)

    def get_optimiser(self):
        return t.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, test_loader, train_loader, n_epochs):
        optimiser = self.get_optimiser()
        period = len(train_loader) // 5
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}")
            avg_loss = 0
            avg_recon_loss = 0
            avg_kld = 0
            for idx, batch in enumerate(train_loader):
                loss, recon_loss, kld = self.train_step(batch)
                loss.backward()
                avg_loss += loss.item()
                avg_recon_loss += recon_loss.item()
                avg_kld += kld.item()
                optimiser.step()
                optimiser.zero_grad()
                if idx % (len(train_loader) // 5) == 0 and idx != 0:
                    print("Training loss", format_loss(avg_loss / period), "Reconstruction loss", format_loss(avg_recon_loss / period), "KLD", avg_kld / period)
                    avg_loss = 0
                    avg_recon_loss = 0
                    avg_kld = 0
        t.save(self.model.state_dict(), "model.pt")


if __name__ == '__main__':
    TRAIN_SUBJECTS = [1, 14, 16, 17, 18, 2, 3, 4, 5, 6]
    TEST_SUBJECTS = [7, 8, 9, 10, 11, 12]
    train_loader, test_loader = get_loaders_just_tt(train_subjects=TRAIN_SUBJECTS, test_subjects=TEST_SUBJECTS)
    model = VAE(latent_dim=75, channels=[12, 32, 128]).cuda()
    model.train()
    print(model)
    trainer = Trainer(model, kld_weight=0.005)  # -> change to ramp up schedule!
    trainer.train(test_loader=test_loader, train_loader=train_loader, n_epochs=30)
