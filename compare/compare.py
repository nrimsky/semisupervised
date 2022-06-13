from supervised.trainer import StandardTrainer
from wav2vec.trainer import W2VTrainer
from timestep_prediction.trainer import TSPTrainer
from simclr.trainer import SimCLRTrainer
from pseudolabelling.pseudolabelling import PseudolabellingTrainer
from pseudolabelling.meta_pseudo_labels import MPLTrainer
from grl.GRL import GRLTrainer
from grl.grl_multisubject import GRLMultisubjectTrainer
from random import shuffle
from datetime import datetime
from helpers import ensure_dir
import os


class ApproachComparer:

    def __init__(self, trainers: dict):
        self.trainers = trainers

    def run(self, unsupervised_subjects, supervised_subjects, identifier=""):
        ensure_dir("results_with_errors", empty=False)
        with open(os.path.join("results_with_errors", f"results_{datetime.now().strftime('%d%m_%H%M')}_{identifier}.txt"), "w") as results_file:
            results_file.write(f"{supervised_subjects=}\n")
            results_file.write(f"{unsupervised_subjects=}\n")
            for approach_name, trainer in self.trainers.items():
                acc, circular_error = trainer.train(unsupervised_subjects=unsupervised_subjects, supervised_subjects=supervised_subjects,
                                    filename=f"{approach_name.replace(' ', '_')}.txt")
                print(approach_name, acc, circular_error)
                results_file.write(f"{approach_name} : {acc},{circular_error}\n")
                if approach_name == "Supervised only" and acc < 0.3:
                    break


if __name__ == '__main__':

    for i in range(5):
        subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
        shuffle(subjects)
        trainers = {
            "Supervised only": StandardTrainer(),
            "Wav2Vec": W2VTrainer(),
            "Future sample detection": TSPTrainer(),
            "SimCLR": SimCLRTrainer(),
            "Pseudolabelling": PseudolabellingTrainer(),
            "Meta Pseudo Labels": MPLTrainer(),
            "GRL": GRLTrainer(),
            # "Subject activity aware GRL": GRLMultisubjectTrainer(optimise_activity=True, optimise_subject=True),
            "Subject aware GRL": GRLMultisubjectTrainer(optimise_activity=False, optimise_subject=True),
            # "Activity aware GRL": GRLMultisubjectTrainer(optimise_activity=True, optimise_subject=False)
        }
        split = int(len(subjects) * 0.8)
        ac = ApproachComparer(trainers=trainers)
        ac.run(unsupervised_subjects=subjects[0:split], supervised_subjects=subjects[split:], identifier="")



