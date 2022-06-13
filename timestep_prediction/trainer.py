import torch as t
from timestep_prediction.model import HybridModel
from helpers import get_files_for_subjects, ClassificationDataset, get_mean_std, evaluate, format_loss, write_and_print, BaseTrainer
from timestep_prediction.timestep_prediction_dataset import TimestepPredictionDataset
from torch.utils import data
from typing import List, Tuple
from random import shuffle


class TSPTrainer(BaseTrainer):

    def __init__(self, n_epochs_unsupervised=15, n_epochs_supervised=10, freeze_body=False, lr_supervised=0.001, lr_unsupervised=0.003):
        self.n_epochs_unsupervised = n_epochs_unsupervised
        self.n_epochs_supervised = n_epochs_supervised
        self.freeze_body = freeze_body
        self.lr_supervised = lr_supervised
        self.lr_unsupervised = lr_unsupervised

    def _train_unsupervised(self, model: HybridModel, self_supervised_train_loader: data.DataLoader, exp_name: str) -> HybridModel:
        criterion = t.nn.CrossEntropyLoss()
        optimiser = t.optim.Adam(model.parameters(), lr=self.lr_unsupervised)
        period = len(self_supervised_train_loader) // 3
        with open(exp_name, "w") as logfile:
            for epoch in range(self.n_epochs_unsupervised):
                write_and_print(f"Epoch {epoch}", logfile)
                avg_loss = 0
                for idx, (batch, labels) in enumerate(self_supervised_train_loader):
                    ref = batch[:, 0].cuda()
                    examples = [batch[:, j].cuda() for j in range(1, batch.shape[1])]
                    classification = model(ref, examples, is_supervised=False)
                    loss = criterion(classification, labels.cuda())
                    avg_loss += loss.item()
                    if idx % period == 0 and idx != 0:
                        write_and_print(f"step {idx}, avg loss = {format_loss(avg_loss / period)}", logfile)
                        avg_loss = 0
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
        return model

    def _train_supervised(self, model: HybridModel, fine_tuning_loader: data.DataLoader, test_loader: data.DataLoader, exp_name: str) -> Tuple[
        HybridModel, float]:
        if self.freeze_body:
            params = model.supervised_head.parameters()
            optimiser = t.optim.Adam(params, lr=self.lr_supervised)
        else:
            optimiser = t.optim.Adam(model.parameters(), lr=self.lr_supervised)
        period = len(fine_tuning_loader) // 3
        criterion = t.nn.CrossEntropyLoss()
        acc = 0
        with open(exp_name, "w") as logfile:
            for epoch in range(self.n_epochs_supervised):
                model.train()
                avg_loss = 0
                for i, (batch, labels) in enumerate(fine_tuning_loader):
                    out = model(batch.cuda(), is_supervised=True)
                    loss = criterion(out, labels.long().cuda())
                    avg_loss += loss.item()
                    if i % period == 0 and i != 0:
                        write_and_print(f"step {i}, avg loss = {format_loss(avg_loss / period)}", logfile)
                        avg_loss = 0
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                model.eval()
                acc = evaluate(model, test_loader)
                write_and_print(f"Epoch {epoch}, Accuracy: {format_loss(acc)}", logfile)
                model.train()
        model.eval()
        acc = evaluate(model, test_loader, use_circ=True)
        return model, acc

    def train(self, unsupervised_subjects: List[int], supervised_subjects: List[int], filename: str) -> float:
        filename_base = filename.split(".")[0]
        unlabelled_files = get_files_for_subjects(unsupervised_subjects, base_dir="../data")
        shuffle(unlabelled_files)
        l = len(unlabelled_files)
        labelled_files = get_files_for_subjects(supervised_subjects, base_dir="../data")
        mean, std = get_mean_std(labelled_files)
        self_supervised_train_dataset = TimestepPredictionDataset(unlabelled_files[:l//2], mean, std, shift_pos=3, shift_neg_std=8)
        test_dataset = ClassificationDataset(unlabelled_files[l//2:], mean, std)
        fine_tuning_dataset = ClassificationDataset(labelled_files, mean, std)
        self_supervised_train_loader = data.DataLoader(self_supervised_train_dataset, batch_size=64, shuffle=True, pin_memory=True,
                                                       drop_last=True)
        fine_tuning_loader = data.DataLoader(fine_tuning_dataset, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
        test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=True)
        t.cuda.empty_cache()
        model = HybridModel(n_examples=3, random_augment=True)
        model.cuda()
        model.train()
        model = self._train_unsupervised(model=model, exp_name=f"pretrained_{filename_base}.txt",
                                         self_supervised_train_loader=self_supervised_train_loader)
        model, acc = self._train_supervised(model=model, exp_name=f"finetuned_{filename_base}.txt", fine_tuning_loader=fine_tuning_loader,
                                            test_loader=test_loader)
        t.save(model.state_dict(), f"{filename_base}.pt")
        model.supervised_head.visualise_embeddings(f"{filename_base}_embeddings")
        return acc


if __name__ == "__main__":
    supervised_subjects = [4, 5, 7, 9]
    unsupervised_subjects = [1, 2, 21, 23]
    trainer = TSPTrainer()
    trainer.train(unsupervised_subjects=unsupervised_subjects, supervised_subjects=supervised_subjects, filename="futdec.txt")
