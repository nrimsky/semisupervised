import torch as t
from helpers import evaluate, format_loss, get_loaders, BaseTrainer, write_and_print
from shared_components import Model
from typing import List


class PseudolabellingTrainer(BaseTrainer):

    def __init__(self, n_epochs=10, n_epochs_finetune=10, threshold=0.18, lr=0.005, finetune_lr=0.0005):
        self.n_epochs = n_epochs
        self.n_epochs_finetune = n_epochs_finetune
        self.threshold = threshold
        self.lr = lr
        self.finetune_lr = finetune_lr

    def train(self, unsupervised_subjects: List[int], supervised_subjects: List[int], filename: str) -> float:
        base_name = filename.split(".")[0]
        unlabelled_loader, labelled_loader, test_loader = get_loaders(unsupervised_subjects, supervised_subjects)
        teacher = Model(channels=(12, 16, 12), head_hidden_dim=64, augment_input=True)  # Use bigger model for teacher
        student = Model(use_hidden=False, augment_input=True)
        teacher.train()
        student.train()
        teacher.cuda()
        student.cuda()
        criterion = t.nn.CrossEntropyLoss()
        teacher_optimiser = t.optim.Adam(teacher.parameters(), lr=self.lr)
        student_optimiser = t.optim.Adam(student.parameters(), lr=self.lr)
        student_finetune = t.optim.Adam(student.parameters(), lr=self.finetune_lr)
        with open(filename, "w") as txtfile:
            write_and_print("Training teacher", txtfile)
            for epoch in range(self.n_epochs):
                avg_loss = 0
                for idx, (batch, labels) in enumerate(labelled_loader):
                    out = teacher(batch.cuda())
                    loss = criterion(out, labels.long().cuda())
                    avg_loss += loss.item()
                    if idx % 100 == 0:
                        write_and_print(f"Teacher Loss: {format_loss(avg_loss / 100)}", txtfile)
                        avg_loss = 0
                    loss.backward()
                    teacher_optimiser.step()
                    teacher_optimiser.zero_grad()
                teacher.eval()
                accuracy = evaluate(teacher, test_loader)
                write_and_print(f"Epoch {epoch}, teacher accuracy: {accuracy}", txtfile)
                teacher.train()
            teacher.eval()
            write_and_print("Training student on data pseudolabelled by teacher", txtfile)
            for epoch in range(self.n_epochs):
                avg_loss = 0
                avg_above_threshold = 0
                for idx, (batch, labels) in enumerate(unlabelled_loader):
                    teacher_out = teacher(batch.cuda())
                    max_probs = t.max(t.softmax(teacher_out, dim=-1), dim=-1).values
                    mask = max_probs > self.threshold
                    if mask.sum() == 0:
                        continue
                    avg_above_threshold += (mask.sum().item() / len(mask))
                    pseudo_labels = t.argmax(teacher_out, dim=-1).long()[mask]
                    out = student((batch[mask]).cuda())
                    loss = criterion(out, pseudo_labels)
                    avg_loss += loss.item()
                    if idx % 100 == 0:
                        write_and_print(f"Student Loss on psuedolabelled data: {format_loss(avg_loss / 100)}", txtfile)
                        write_and_print(f"Average fraction of pseudolabels with required confidence: {avg_above_threshold / 100}", txtfile)
                        avg_loss = 0
                        avg_above_threshold = 0
                    loss.backward()
                    student_optimiser.step()
                    student_optimiser.zero_grad()
                student.eval()
                accuracy = evaluate(student, test_loader)
                write_and_print(f"Epoch {epoch}, student trained on pseudolabelled data accuracy: {accuracy}", txtfile)
                student.train()
            write_and_print("Fine tuning student", txtfile)
            accuracy = 0
            for epoch in range(self.n_epochs_finetune):
                avg_loss = 0
                for idx, (batch, labels) in enumerate(labelled_loader):
                    out = student(batch.cuda())
                    loss = criterion(out, labels.long().cuda())
                    avg_loss += loss.item()
                    if idx % 100 == 0:
                        write_and_print(f"Student Loss on labelled data: {format_loss(avg_loss / 100)}", txtfile)
                        avg_loss = 0
                    loss.backward()
                    student_finetune.step()
                    student_finetune.zero_grad()
                student.eval()
                accuracy = evaluate(student, test_loader)
                write_and_print(f"Epoch {epoch}, finetuned student accuracy: {accuracy}", txtfile)
                student.train()
            student.eval()
            accuracy = evaluate(student, test_loader, use_circ=True)
            student.head.visualise_embeddings(f"student_embeddings_{base_name}")
            teacher.head.visualise_embeddings(f"teacher_embeddings_{base_name}")
            return accuracy


if __name__ == "__main__":
    supervised_subjects = [1, 2]
    unsupervised_subjects = [3, 4]
    trainer = PseudolabellingTrainer()
    trainer.train(unsupervised_subjects=unsupervised_subjects, supervised_subjects=supervised_subjects, filename="pseudolabelling.txt")

