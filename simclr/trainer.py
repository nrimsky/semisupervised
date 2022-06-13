import torch as t
from helpers import ClassificationDataset, get_files_for_subjects, evaluate, format_loss, BaseTrainer, get_mean_std, write_and_print
from shared_components import ClassificationHead
from torch.utils.data import Dataset
from simclr.SimCLR import ConvNetSimCLR, ContrastiveLearningDataset, SimCLR
from typing import Tuple, List
from random import shuffle


class SimCLRTrainer(BaseTrainer):

    def __init__(self, n_epochs_unsupervised=2, n_epochs_supervised=10, lr_unsupervised=0.005, lr_supervised=0.001):
        self.n_epochs_unsupervised = n_epochs_unsupervised
        self.n_epochs_supervised = n_epochs_supervised
        self.lr_unsupervised = lr_unsupervised
        self.lr_supervised = lr_supervised


    @staticmethod
    def make_supervised_model(unsupervised_model: ConvNetSimCLR):
        model = ConvNetSimCLR()
        model.eval()
        model.load_state_dict(unsupervised_model.state_dict())
        model.head = ClassificationHead(use_hidden=True).cuda()
        model.cuda()
        model.train()
        print("Model set up")
        return model

    def unsupervised_pretrain(self, files, mean, std, logfilename, n_views=3, batch_size=64, wd=1e-4, temperature=0.07) -> ConvNetSimCLR:
        dataset = ContrastiveLearningDataset(files, mean, std)
        train_dataset = dataset.get_dataset(n_views)
        train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        model = ConvNetSimCLR()
        optimizer = t.optim.Adam(model.parameters(), self.lr_unsupervised, weight_decay=wd)
        scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
        simclr = SimCLR(model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        device="cuda",
                        batch_size=batch_size,
                        n_views=n_views,
                        temperature=temperature,
                        epochs=self.n_epochs_unsupervised,
                        filename=logfilename)
        model = simclr.train(train_loader)
        return model

    def finetune(self, pretrained_model, train_loader, test_loader, filename, freeze_body=False) -> Tuple[ConvNetSimCLR, float]:
        if freeze_body:
            optimizer = t.optim.Adam(pretrained_model.head.parameters(), lr=self.lr_supervised)  # Only update head!
        else:
            optimizer = t.optim.Adam(pretrained_model.parameters(), lr=self.lr_supervised)
        criterion = t.nn.CrossEntropyLoss()
        acc = 0
        with open(filename, "w") as txtfile:
            for epoch_counter in range(self.n_epochs_supervised):
                write_and_print(f"Epoch: {epoch_counter}", txtfile)
                avg_loss = 0
                for idx, (batch, labels) in enumerate(train_loader):
                    batch = batch.to("cuda", dtype=t.float)
                    labels = labels.to("cuda", dtype=t.long)
                    logits = pretrained_model(batch)
                    loss = criterion(logits, labels)
                    avg_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if idx % 300 == 0:
                        write_and_print(f"Loss: {format_loss(avg_loss / 300)}", txtfile)
                        avg_loss = 0
                pretrained_model.eval()
                acc = evaluate(pretrained_model, test_loader)
                pretrained_model.train()
                write_and_print(f"Accuracy: {acc}", txtfile)
            pretrained_model.eval()
            acc = evaluate(pretrained_model, test_loader, use_circ=True)
        return pretrained_model, acc

    def train(self, unsupervised_subjects: List[int], supervised_subjects: List[int], filename: str) -> float:
        base_name = filename.split(".")[0]
        mean, std = get_mean_std(get_files_for_subjects(supervised_subjects, base_dir="../data"))
        unsup_files = get_files_for_subjects(unsupervised_subjects, base_dir="../data")
        shuffle(unsup_files)
        l = len(unsup_files)
        sup_files = get_files_for_subjects(supervised_subjects, base_dir="../data")
        pretrained_model = self.unsupervised_pretrain(files=unsup_files[:l//2], mean=mean, std=std, logfilename="pretrain_"+filename)
        supervised_dataset = ClassificationDataset(sup_files, mean, std)
        test_dataset = ClassificationDataset(unsup_files[l//2:], mean, std)
        train_loader = t.utils.data.DataLoader(supervised_dataset, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
        test_loader = t.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=True)
        model = self.make_supervised_model(pretrained_model)
        model, accuracy = self.finetune(pretrained_model=model, train_loader=train_loader, test_loader=test_loader, filename="fine_tune_" + filename)
        t.save(model.state_dict(), f"{base_name}_finetuned.pt")
        model.eval()
        model.head.visualise_embeddings(f"{base_name}_embeddings")
        return accuracy


if __name__ == "__main__":
    supervised_subjects = [1, 2]
    unsupervised_subjects = [3, 4]
    trainer = SimCLRTrainer()
    trainer.train(unsupervised_subjects=unsupervised_subjects, supervised_subjects=supervised_subjects, filename="SimCLR.txt")

