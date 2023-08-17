"""Curriculum Semantic-aware Contrastive Learning."""

import os
import pickle

import numpy as np

from transformers import BartTokenizer

from dataset import ZuCo, build_CSCL_maps


class CSCL:
    """Contrastive Self-Supervised Curriculum Learning (CSCL).

    This class implements a Contrastive Self-Supervised Curriculum Learning framework.
    Given EEG signals, sentences, and curriculum levels, it generates contrastive triplets
    to guide the learning process.

    Args:
    ----
        sentences (list): A set of all sentences S.
        fs (dict): A mapping from sentence Si to a set of EEG signals ESi.
        fp (dict): A mapping from subject pi to a set of EEG signals Epi.

    """

    def __init__(self, sentences, fs, fp):
        self.sentences = sentences
        self.fs = fs
        self.fp = fp

    def C_SCL(self, Ei, pi, Si, curr_level):
        """Create a contrastive triplet.

        C_SCL(Ei, pi, Si, curr_level):
            Create a contrastive triplet composed of a positive sample E_plus_i_curr_level
            and a negative sample E_minus_i_curr_level.
        """
        # Positive sample
        E_plus_i = self.fs[Si] - Ei
        E_plus_i_sorted_desc = self.cur_cri(E_plus_i, order='descend')
        curriculums = self.cur_lev(E_plus_i_sorted_desc)
        E_plus_i_curr_level = self.cur_sche(curriculums, curr_level)

        # Negative sample
        S_minus_i = list(set(self.sentences) - set([Si]))
        E_minus_i = np.concatenate([self.fp[p] for p in self.fp.keys() if p != pi])
        E_minus_i_sorted_asc = self.cur_cri(E_minus_i, order='ascend')
        curriculums = self.cur_lev(E_minus_i_sorted_asc)
        E_minus_i_curr_level = self.cur_sche(curriculums, curr_level)

        return Ei, E_plus_i_curr_level, E_minus_i_curr_level

    def cur_cri(self, E, order):
        """Curriculum criterion.

        cur_cri(E, order):
            Calculate the curriculum criterion, sorting EEG signals based on similarity.
        """
        sims = []
        for Ej in E:
            # simj = cosine_similarity(E, Ej)
            sims.append(simj)
        indices = np.argsort(sims, order=order)
        return E[indices]

    def cur_lev(self, E):
        """Curriculum level.

        cur_lev(E):
            Determine curriculum levels by dividing EEG signals into easy, medium, and hard subsets.
        """
        length = len(E)
        step = length // 3
        E_easy = E[:step]
        E_medium = E[step:2*step]
        E_hard = E[2*step:]
        return [E_easy, E_medium, E_hard]

    def cur_sche(self, curriculums, curr_level):
        """Curriculum scheduler.

        cur_sche(curriculums, curr_level):
            Select EEG signals based on the current curriculum level.
        """
        E_select = curriculums[curr_level]
        selected_E = np.random.choice(E_select)
        return selected_E


if __name__ == "__main__":

    # dataset splits
    whole_dataset_dicts = []

    dataset_path_task1 = os.path.join(
        '../', 'dataset', 'ZuCo',
        'task1-SR', 'pickle', 'task1-SR-dataset.pickle'
        )

    dataset_path_task2 = os.path.join(
        '../', 'dataset', 'ZuCo',
        'task2-NR', 'pickle', 'task2-NR-dataset.pickle'
        )

    dataset_path_task2_v2 = os.path.join(
        '../', 'dataset', 'ZuCo',
        'task2-NR-2.0', 'pickle', 'task2-NR-2.0-dataset.pickle'
        )

    whole_dataset_dicts = []
    for t in [dataset_path_task1, dataset_path_task2, dataset_path_task2_v2]:
        with open(t, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    subject_choice = 'ALL'
    eeg_type_choice = 'GD'
    bands_choice = 'ALL'
    dataset_setting = 'unique_sent'

    train_data = ZuCo(
        whole_dataset_dicts,
        'train',
        tokenizer,
        subject=subject_choice,
        eeg_type=eeg_type_choice,
        bands=bands_choice,
        setting=dataset_setting
        )

    fs, fp, S = build_CSCL_maps(train_data)
    # cscl = CSCL(fp, fs, S)
    # triplet cscl.get_triplet(EGG, subject, sentence, curriculum level) or
    # or
    # triplet = cscl(EGG, subject, sentence, curriculum level)
