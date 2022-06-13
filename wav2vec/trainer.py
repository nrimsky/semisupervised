from helpers import evaluate, BaseTrainer, write_and_print, count_parameters
from typing import List
from wav2vec.dataset import get_loaders
import torch as t
from wav2vec.model import SelfSupervisedModel, SupervisedModel


class W2VTrainer(BaseTrainer):

    def __init__(self,
                 lr=0.003,
                 n_epochs=15,
                 n_timesteps_z=3,
                 n_sections_context=10,
                 n_step_specific=10,
                 context_output_size=12,
                 encoder_output_size=6):
        """
        :param lr: Learning rate
        :param n_epochs: Number of training epochs
        :param n_timesteps_z: Number of timesteps encoded with encoder network
        :param n_sections_context: Number of encoded sections of timesteps provided to context network
        :param n_step_specific: Number of step specific transforms learnt
        :param context_output_size: Output dimensionality of context network
        :param encoder_output_size: Output dimensionality of encoder network
        """

        self.lr = lr
        self.n_epochs = n_epochs
        self.n_timesteps_z = n_timesteps_z
        self.n_sections_context = n_sections_context
        self.n_step_specific = n_step_specific
        self.context_output_size = context_output_size
        self.encoder_output_size = encoder_output_size

    def train_self_supervised(self, us_loader, save_to, write_file):
        model = SelfSupervisedModel(n_timesteps_z=self.n_timesteps_z,
                                    n_sections_context=self.n_sections_context,
                                    context_output_size=self.context_output_size,
                                    encoder_output_size=self.encoder_output_size,
                                    n_step_specific=self.n_step_specific)
        print(count_parameters(SupervisedModel(model)))
        exit()
        model = model.cuda()
        model.train()
        optimiser = t.optim.Adam(model.parameters(), lr=self.lr)
        criterion = t.nn.CrossEntropyLoss()
        period = len(us_loader) // 10
        for epoch in range(self.n_epochs):
            write_and_print(f"Epoch {epoch + 1}", write_file)
            avg_loss = 0
            for idx, (batch, _) in enumerate(us_loader):
                batch = batch.cuda()
                out = model(batch)
                batch_size = len(batch)
                labels = t.repeat_interleave(t.arange(self.n_step_specific), batch_size).cuda().long()
                loss = criterion(out, labels)
                avg_loss += loss.item()
                if idx % period == 0 and idx != 0:
                    write_and_print(f"Loss = {avg_loss / period}", write_file)
                    avg_loss = 0
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
        t.save(model, save_to)

    def train_supervised(self, s_loader, test_loader, load_filename, save_filename, write_file, name_base) -> float:
        self_supervised = t.load(load_filename)
        self_supervised.eval()
        model = SupervisedModel(self_supervised)
        model.train()
        model.cuda()
        optimiser = t.optim.Adam(model.parameters(), lr=self.lr/6)
        criterion = t.nn.CrossEntropyLoss()
        period = len(s_loader) // 10
        acc = 0
        for epoch in range(self.n_epochs):
            write_and_print(f"Epoch {epoch + 1}", write_file)
            avg_loss = 0
            for idx, (batch, labels) in enumerate(s_loader):
                batch = batch.cuda()
                labels = labels.long().cuda()
                out = model(batch)
                loss = criterion(out, labels)
                avg_loss += loss.item()
                if idx % period == 0 and idx != 0:
                    write_and_print(f"Loss = {avg_loss / period}", write_file)
                    avg_loss = 0
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
            model.eval()
            acc = evaluate(model, test_loader)
            write_and_print(f"Test accuracy = {acc}", write_file)
            model.train()
        model.head.visualise_embeddings(f"{name_base}_embeddings")
        t.save(model, save_filename)
        model.eval()
        acc = evaluate(model, test_loader, use_circ=True)
        return acc

    def train(self, unsupervised_subjects: List[int], supervised_subjects: List[int], filename: str) -> float:
        filename_base = filename.split(".")[0]
        us_filename = f'pretrained_{filename_base}.pt'
        s_filename = f'final_{filename_base}.pt'
        timeblock_size = (self.n_step_specific + self.n_sections_context) * self.n_timesteps_z
        with open(filename, "w") as write_file:
            unsuper_train_loader, super_train_loader, test_loader = get_loaders(unlabelled=unsupervised_subjects,
                                                                                labelled=supervised_subjects,
                                                                                n_timesteps_super=self.n_sections_context * self.n_timesteps_z,
                                                                                n_timesteps_unsuper=timeblock_size,
                                                                                step=10)
            self.train_self_supervised(us_loader=unsuper_train_loader,
                                       save_to=us_filename,
                                       write_file=write_file)
            acc = self.train_supervised(s_loader=super_train_loader,
                                        test_loader=test_loader,
                                        load_filename=us_filename,
                                        save_filename=s_filename,
                                        write_file=write_file,
                                        name_base=filename_base)
        return acc


if __name__ == '__main__':
    trainer = W2VTrainer()
    supervised_subjects = [1, 2]
    unsupervised_subjects = [3, 4]
    trainer.train(supervised_subjects=supervised_subjects, unsupervised_subjects=unsupervised_subjects, filename="wav2vec.txt")
