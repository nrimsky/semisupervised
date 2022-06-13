from shared_components import Model
from helpers import evaluate, get_loaders_just_tt, BaseTrainer, write_and_print
import torch as t
from typing import List


class StandardTrainer(BaseTrainer):

    def __init__(self, lr=0.003, n_epochs=10):
        self.lr = lr
        self.n_epochs = n_epochs

    def train(self, unsupervised_subjects: List[int], supervised_subjects: List[int], filename: str) -> float:
        with open(filename, "w") as txtfile:
            write_and_print(f"Unsupervised Subjects: {unsupervised_subjects}", txtfile)
            write_and_print(f"Supervised Subjects: {supervised_subjects}", txtfile)
            write_and_print(f"LR: {self.lr}", txtfile)
            model = Model(use_dropout=False, augment_input=True, use_hidden=True)
            train_loader, test_loader = get_loaders_just_tt(train_subjects=supervised_subjects, test_subjects=unsupervised_subjects)
            t.cuda.empty_cache()
            model.train()
            model.cuda()
            optimiser = t.optim.Adam(model.parameters(), lr=self.lr)
            criterion = t.nn.CrossEntropyLoss()
            l = len(train_loader)
            period = l // 10
            for epoch in range(self.n_epochs):
                avg_loss = 0
                write_and_print(f"Epoch {epoch + 1}", txtfile)
                for i, (_batch, _label) in enumerate(train_loader):
                    label = _label.long().cuda()
                    batch = _batch.cuda()
                    out = model(batch)
                    loss = criterion(out, label)
                    avg_loss += loss.item()
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                    if i % period == 0 and i != 0:
                        write_and_print(f"Loss = {avg_loss / period}", txtfile)
                        avg_loss = 0
                model.eval()
                with t.no_grad():
                    acc = evaluate(model, test_loader)
                model.train()
                write_and_print(f"Test accuracy = {acc}", txtfile)

            model.eval()
            acc = evaluate(model, test_loader, use_circ=True)
            t.save(model.state_dict(), f"{filename.split('.')[0]}.pt")
            model.head.visualise_embeddings(f"{filename.split('.')[0]}_embeddings")
            return acc


if __name__ == "__main__":
    supervised_subjects = [1, 2]
    unsupervised_subjects = [3, 4]
    trainer = StandardTrainer()
    trainer.train(unsupervised_subjects=unsupervised_subjects, supervised_subjects=supervised_subjects, filename="supervised.txt")