import torch as t
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from helpers import cyclical_err, get_files_for_subjects, get_mean_std, ensure_dir, SubjectActivityAwareDataset, format_loss, BaseTrainer, write_and_print, augment
from shared_components import Body, ClassificationHead, ProjectionHead
import torch.optim as optim
from pathlib import Path
import os
from grl.GRL import ReverseLayer
from typing import List


class Model(t.nn.Module):

    def __init__(self, num_subjects, num_activities, augment_input=True):
        super().__init__()
        self.body = Body()
        self.stage_classification_head = ClassificationHead(use_hidden=False)
        self.activity_classification_head = ProjectionHead(output_dim=num_activities, hidden_dim=32)
        self.subject_classification_head = ProjectionHead(output_dim=num_subjects, hidden_dim=32)
        self.augment = augment_input

    def forward(self, input_data, alpha):
        if self.augment and self.training:
            with t.no_grad():
                input_data = augment(input_data)
        feature = self.body(input_data)
        reverse_feature = ReverseLayer.apply(feature, alpha)
        stage_label = self.stage_classification_head(feature)
        activity_label = self.activity_classification_head(reverse_feature)
        subject_label = self.subject_classification_head(reverse_feature)
        return stage_label, activity_label, subject_label


class GRLMultisubjectTrainer(BaseTrainer):

    def __init__(self, n_steps=20000, lr=0.002, optimise_activity=True, optimise_subject=True):
        self.n_steps = n_steps
        self.lr = lr
        self.optimise_activity = optimise_activity
        self.optimise_subject = optimise_subject

    @staticmethod
    def evaluate(model: Model, dataloader: DataLoader, use_c_err = False):
        n_total = 0
        n_correct_stage = 0
        n_correct_activity = 0
        n_correct_subject = 0
        c_err = 0
        n = 0
        for i, (data, stage_labels, activity_labels, subject_labels) in enumerate(dataloader):
            batch_size = len(stage_labels)
            data = data.cuda().float()
            stage_output, activity_output, subject_output = model(input_data=data, alpha=1)
            stage_pred = stage_output.data.max(1)[1].cpu()
            activity_pred = activity_output.data.max(1)[1].cpu()
            subject_pred = subject_output.data.max(1)[1].cpu()
            n_correct_stage += stage_pred.eq(stage_labels).sum().item()
            n_correct_activity += activity_pred.eq(activity_labels).sum().item()
            n_correct_subject += subject_pred.eq(subject_labels).cpu().sum().item()
            c_err += float(cyclical_err(pred=stage_pred, labels=stage_labels))
            n_total += batch_size
            n += 1
        accu_stage = n_correct_stage * 1.0 / n_total
        accu_activity = n_correct_activity * 1.0 / n_total
        accu_subject = n_correct_subject * 1.0 / n_total
        avg_c_err = c_err * 1.0 / n
        if use_c_err:
            return accu_stage, avg_c_err
        return accu_stage, accu_activity, accu_subject

    def train(self, unsupervised_subjects: List[int], supervised_subjects: List[int], filename: str) -> float:
        base_name = filename.split(".")[0]
        ensure_dir(base_name)
        base_dir = os.path.join(Path(os.getcwd()).parent.absolute(), "data")
        source_domain_files = get_files_for_subjects(supervised_subjects, base_dir=base_dir)
        target_domain_files = get_files_for_subjects(unsupervised_subjects, base_dir=base_dir)
        mean, std = get_mean_std(source_domain_files)

        all_subjects = unsupervised_subjects + supervised_subjects
        n_subjects = len(all_subjects)
        subject_labels = {
            int(s): i for i, s in enumerate(all_subjects)
        }

        dataset_source = SubjectActivityAwareDataset(
            source_domain_files, mean, std, relabel_dict=subject_labels
        )

        dataloader_source = t.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=64,
            shuffle=True,
            drop_last=True
        )

        dataset_target = SubjectActivityAwareDataset(
            target_domain_files, mean, std, relabel_dict=subject_labels
        )

        unlabelled_train_idx, test_idx = train_test_split(list(range(len(dataset_target))), test_size=0.5)

        dataloader_target = t.utils.data.DataLoader(
            dataset=Subset(dataset_target, unlabelled_train_idx),
            batch_size=64,
            shuffle=True,
            drop_last=True
        )

        dataloader_test = t.utils.data.DataLoader(
            dataset=Subset(dataset_target, test_idx),
            batch_size=64,
            shuffle=False,
            drop_last=True
        )

        # load model

        model = Model(num_activities=3, num_subjects=n_subjects)
        model.train()

        # setup optimizer

        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        criterion = t.nn.CrossEntropyLoss()

        model = model.cuda()

        target_iter = iter(dataloader_target)
        source_iter = iter(dataloader_source)

        # training

        avg_err_label_source = 0

        avg_err_subject_source = 0
        avg_err_activity_source = 0

        avg_err_subject_target = 0
        avg_err_activity_target = 0

        with open(filename, "w") as txtfile:

            for step in range(self.n_steps):
                try:
                    batch_s, label_s, subject_s, activity_s = next(source_iter)
                except StopIteration:
                    source_iter = iter(dataloader_source)
                    batch_s, label_s, subject_s, activity_s = next(source_iter)
                batch_s, label_s, subject_s, activity_s = batch_s.cuda(), label_s.cuda(), subject_s.cuda(), activity_s.cuda()
                try:
                    batch_t, _, subject_t, activity_t = next(target_iter)
                except StopIteration:
                    target_iter = iter(dataloader_target)
                    batch_t, _, subject_t, activity_t = next(target_iter)
                batch_t, subject_t, activity_t = batch_t.cuda(), subject_t.cuda(), activity_t.cuda()

                p = step / self.n_steps
                alpha = 0.03 * p

                source_stage_output, source_activity_output, source_subject_output = model(input_data=batch_s, alpha=alpha / 10)

                source_stage_error = criterion(source_stage_output, label_s)
                if self.optimise_activity:
                    source_activity_error = criterion(source_activity_output, activity_s)
                else:
                    source_activity_error = t.tensor(0)
                if self.optimise_subject:
                    source_subject_error = criterion(source_subject_output, subject_s)
                else:
                    source_subject_error = t.tensor(0)

                _, target_activity_output, target_subject_output = model(input_data=batch_t, alpha=alpha / 10)

                if self.optimise_activity:
                    target_activity_error = criterion(target_activity_output, activity_t)
                else:
                    target_activity_error = t.tensor(0)

                if self.optimise_subject:
                    target_subject_error = criterion(target_subject_output, subject_t)
                else:
                    target_subject_error = t.tensor(0)

                err = source_stage_error + source_activity_error + source_subject_error + target_activity_error + target_subject_error
                err.backward()
                optimizer.step()
                optimizer.zero_grad()

                avg_err_label_source += source_stage_error.item()

                avg_err_subject_source += source_subject_error.item()
                avg_err_activity_source += source_activity_error.item()

                avg_err_subject_target += target_subject_error.item()
                avg_err_activity_target += target_activity_error.item()

                if step % (self.n_steps // 30) == 0 and step != 0:
                    write_and_print(f'step: {step} : '
                                    f'avg_err_label_source: {format_loss(avg_err_label_source / (self.n_steps // 30))}, '
                                    f'avg_err_subject_source: {format_loss(avg_err_subject_source / (self.n_steps // 30))}, '
                                    f'avg_err_activity_source: {format_loss(avg_err_activity_source / (self.n_steps // 30))} '
                                    f'avg_err_subject_target: {format_loss(avg_err_subject_target / (self.n_steps // 30))} '
                                    f'avg_err_activity_target: {format_loss(avg_err_activity_target / (self.n_steps // 30))}', txtfile)
                    avg_err_label_source = 0
                    avg_err_subject_source = 0
                    avg_err_activity_source = 0
                    avg_err_subject_target = 0
                    avg_err_activity_target = 0
                if step % (self.n_steps // 10) == 0 and step != 0:
                    model.eval()
                    accuracy = self.evaluate(model, dataloader_test)
                    write_and_print(f"Accuracy: {accuracy}", txtfile)
                    model.train()
            t.save(model, f'{base_name}/model.pth')
            model.stage_classification_head.visualise_embeddings(f"{base_name}_embeddings")
            model.eval()
            accuracy = self.evaluate(model, dataloader_test, use_c_err=True)
            write_and_print(f"Accuracy: {accuracy}", txtfile)
            print('done')
            return accuracy


if __name__ == '__main__':
    supervised_subjects = [18, 12, 25, 11, 3, 29, 7, 4, 26, 21, 16]
    unsupervised_subjects = [17, 23, 9, 20, 10, 15, 1, 19, 2, 5, 14, 28]
    trainer = GRLMultisubjectTrainer()
    trainer.train(unsupervised_subjects=unsupervised_subjects, supervised_subjects=supervised_subjects, filename="grl_sub_ac.txt")