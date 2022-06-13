from torch.autograd import Function
import torch as t
from helpers import ensure_dir, format_loss, BaseTrainer, write_and_print, augment, get_loaders, evaluate
from shared_components import Body, ClassificationHead, ProjectionHead
import torch.optim as optim
from typing import List


class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Model(t.nn.Module):

    def __init__(self, augment_input=True):
        super().__init__()
        self.body = Body()
        self.class_classification_head = ClassificationHead(use_hidden=False)
        self.domain_classification_head = ProjectionHead(output_dim=2)
        self.augment = augment_input

    def forward(self, input_data, alpha=0):
        if self.augment and self.training:
            with t.no_grad():
                input_data = augment(input_data)
        feature = self.body(input_data)
        reverse_feature = ReverseLayer.apply(feature, alpha)
        class_output = self.class_classification_head(feature)
        domain_output = self.domain_classification_head(reverse_feature)
        return class_output, domain_output


class GRLTrainer(BaseTrainer):

    def __init__(self, n_steps=10000, lr=0.002):
        self.n_steps = n_steps
        self.lr = lr

    def train(self, unsupervised_subjects: List[int], supervised_subjects: List[int], filename: str) -> float:
        base_name = filename.split(".")[0]
        ensure_dir(base_name)

        dataloader_target, dataloader_source, test_loader = get_loaders(unlabelled_subjects=unsupervised_subjects, labelled_subjects=supervised_subjects)

        # load model

        model = Model()
        model.train()

        # setup optimizer

        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        criterion = t.nn.CrossEntropyLoss()

        model = model.cuda()

        target_iter = iter(dataloader_target)
        source_iter = iter(dataloader_source)

        # training

        avg_err_s_label = 0
        avg_err_s_domain = 0
        avg_err_t_domain = 0


        with open(filename, "w") as txtfile:
            for step in range(self.n_steps):
                try:
                    batch_s, labels_s = next(source_iter)
                except StopIteration:
                    source_iter = iter(dataloader_source)
                    batch_s, labels_s = next(source_iter)
                try:
                    batch_t, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(dataloader_target)
                    batch_t, _ = next(target_iter)

                p = step / self.n_steps
                alpha = 0.03 * p

                batch_size = len(labels_s)
                s_data = batch_s.cuda().float()
                s_label = labels_s.cuda().long()
                domain_label = t.zeros(batch_size, device='cuda', dtype=t.long)

                class_output, domain_output = model(input_data=s_data, alpha=alpha)
                err_s_label = criterion(class_output, s_label)
                err_s_domain = criterion(domain_output, domain_label)

                t_data = batch_t.cuda().float()
                domain_label = t.ones(batch_size, device='cuda', dtype=t.long)
                _, domain_output = model(input_data=t_data, alpha=alpha)
                err_t_domain = criterion(domain_output, domain_label)
                err = err_t_domain + err_s_domain + err_s_label
                err.backward()
                optimizer.step()
                optimizer.zero_grad()

                avg_err_s_label += err_s_label.cpu().item()
                avg_err_s_domain += err_s_domain.cpu().item()
                avg_err_t_domain += err_t_domain.cpu().item()

                if step % (self.n_steps // 30) == 0 and step != 0:
                    write_and_print(f'step: {step} : '
                                    f'err_s_label: {format_loss(avg_err_s_label / (self.n_steps // 30))}, '
                                    f'err_s_domain: {format_loss(avg_err_s_domain / (self.n_steps // 30))}, '
                                    f'err_t_domain: {format_loss(avg_err_t_domain / (self.n_steps // 30))}', txtfile)
                    avg_err_s_label = 0
                    avg_err_s_domain = 0
                    avg_err_t_domain = 0
                if step % (self.n_steps // 10) == 0 and step != 0:
                    model.eval()
                    acc = evaluate(model=model, data_loader=test_loader)
                    write_and_print(f"Accuracy: {acc}", txtfile)
                    model.train()
            model.eval()
            acc = evaluate(model=model, data_loader=test_loader, use_circ=True)
            write_and_print(f"Accuracy: {acc}", txtfile)
            t.save(model, f'{base_name}/model.pth')
            model.class_classification_head.visualise_embeddings(f"{base_name}_embeddings")
            print('done')

        return acc


if __name__ == '__main__':
    supervised_subjects = [1, 2]
    unsupervised_subjects = [3, 4]
    trainer = GRLTrainer()
    trainer.train(unsupervised_subjects=unsupervised_subjects, supervised_subjects=supervised_subjects, filename="grl.txt")
