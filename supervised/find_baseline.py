from shared_components import Model
from helpers import ensure_dir, evaluate, write_and_print, get_loaders_just_tt
import glob
from datetime import datetime
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import os
from globals import WINDOW_SIZE, DOWN_SAMPLE, N_CHANNELS_CONV, CONV_SIZE, POOL_SIZE


if __name__ == "__main__":

    ensure_dir("benchmarking", empty=False)
    ensure_dir("logs2", empty=False)
    ensure_dir("embeddings", empty=False)

    SUBJECTS = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 28, 29]
    LR = 3e-3
    N_EPOCHS = 10
    #
    for subject in SUBJECTS:
        print(f"Testing on subject {subject}")
        with open(f"logs2/{subject}_performance.txt", "w") as txtfile:
            write_and_print(f"window size: {WINDOW_SIZE}, downsample: {DOWN_SAMPLE}, channels: {N_CHANNELS_CONV}, conv_size: {CONV_SIZE}, pool_size: {POOL_SIZE}", txtfile)
            model = Model(use_dropout=False, augment_input=True, use_hidden=True)
            train_subjects = [s for s in SUBJECTS if s != subject]
            test_subject = [subject]
            train_loader, test_loader = get_loaders_just_tt(train_subjects, test_subject)
            t.cuda.empty_cache()
            model.train()
            model.cuda()
            optimiser = t.optim.Adam(model.parameters(), lr=LR)
            criterion = t.nn.CrossEntropyLoss()
            l = len(train_loader)
            period = l//10
            for epoch in range(N_EPOCHS):
                write_and_print(f"Epoch: {epoch}", txtfile)
                for i, (_batch, _label) in enumerate(train_loader):
                    label = _label.long().cuda()
                    batch = _batch.cuda()
                    out = model(batch)
                    loss = criterion(out, label)
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                    if i % period == 0 and i != 0:
                        model.eval()
                        acc, c_err = evaluate(model, test_loader, use_circ=True)
                        write_and_print(f"Batch {i}: {acc},{c_err}", txtfile)
                        model.train()
        model.head.visualise_embeddings(os.path.join("embeddings", f"embeddings_{subject}"))

    files = glob.glob("logs2/*")
    data = np.zeros(max(SUBJECTS) + 1)
    data_c_err = np.zeros(max(SUBJECTS) + 1)

    for l in glob.glob("logs2/*"):
        with open(l, "r") as f:
            content = f.read()
            res = content.split(" ")[-1]
            acc, c_err = res.split(",")
            acc = float(acc)
            c_err = float(c_err)
            subject = int(l.split("_")[0].split("\\")[-1])
            data[subject] = acc
            data_c_err[subject] = c_err

    plt.figure(figsize=(7, 7))
    filled_data = data[data != 0]
    mean = filled_data.mean()
    plt.xlim([0, len(filled_data) + 1])
    plt.ylim([0, 1])
    plt.yticks([z / 10 for z in range(11)])
    plt.xlabel("Subject ID")
    plt.ylabel(f"Accuracy after {N_EPOCHS} epochs")
    plt.title("Accuracy of model for each possible split of test subject/ training subjects")
    plt.axhline(y=mean, color='r', linestyle='-', label="Mean accuracy")
    plt.axhline(y=1 / 13, color='g', linestyle='-', label="Random accuracy")
    plt.legend()
    labels = [str(i) for i in range(1, len(filled_data) + 1)]
    plt.bar(range(1, len(filled_data) + 1), list(filled_data), tick_label=labels)
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    plt.savefig(os.path.join("benchmarking", f"testing_splits_{dt_string}.png"))

    plt.figure(figsize=(7, 7))
    filled_data = data_c_err[data_c_err != 0]
    mean = filled_data.mean()
    plt.xlim([0, len(filled_data) + 1])
    plt.ylim([0, 3])
    plt.yticks([z*0.5 for z in range(7)])
    plt.xlabel("Subject ID")
    plt.ylabel(f"Error after {N_EPOCHS} epochs")
    plt.title("Circular error of model for each possible split of test subject/ training subjects")
    plt.axhline(y=mean, color='r', linestyle='-', label="Mean error")
    plt.legend()
    labels = [str(i) for i in range(1, len(filled_data) + 1)]
    plt.bar(range(1, len(filled_data) + 1), list(filled_data), tick_label=labels)
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    plt.savefig(os.path.join("benchmarking", f"c_err_testing_splits_{dt_string}.png"))
