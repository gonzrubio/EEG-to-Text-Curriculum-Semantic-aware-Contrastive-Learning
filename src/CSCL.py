"""Curriculum Semantic-aware Contrastive Learning."""

import os
import pickle

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from dataset import ZuCo, build_CSCL_maps


class CSCL:
    """Contrastive Self-Supervised Curriculum Learning (CSCL).

    This class implements a Contrastive Self-Supervised Curriculum Learning framework.
    Given EEG signals, sentences, and curriculum levels, it generates contrastive triplets
    to guide the learning process.

    Args:
    ----
        S (set): The set of all sentences.
        fs (dict): A mapping from sentence Si to a set of EEG signals ESi.
        fp (dict): A mapping from subject pi to a set of EEG signals Epi.

    """

    def __init__(self, fs, fp, S):
        self.fs = fs
        self.fp = fp
        self.S = S

    def get_triplet(self, Ei, pi, Si, curr_level):
        """Create a contrastive triplet."""
        # Positive sample
        # E_plus_i = self.fs[Si] - Ei
        E_plus_i = self.fs[Si[0]]
        E_plus_i_sorted_desc = self.cur_cri(Ei[0], E_plus_i, order='descend')
        curriculums = self.cur_lev(E_plus_i_sorted_desc)
        E_plus_i_curr_level = self.cur_sche(curriculums, curr_level)

        # Negative sample
        S_minus_i = list(set(self.sentences) - set([Si]))
        E_minus_i = np.concatenate([self.fp[p] for p in self.fp.keys() if p != pi])
        E_minus_i_sorted_asc = self.cur_cri(E_minus_i, order='ascend')
        curriculums = self.cur_lev(E_minus_i_sorted_asc)
        E_minus_i_curr_level = self.cur_sche(curriculums, curr_level)

        return Ei, E_plus_i_curr_level, E_minus_i_curr_level

    def cur_cri(self, Ei, E, order):
        """Curriculum criterion - sort the EEG signals based on similarity."""
        sims = []
        for (Ej, _) in E:
            # print(Ei.shape, Ej.shape)
            simj = F.cosine_similarity(
                Ei.sum(dim=0) / Ei[:, 0].count_nonzero(),
                Ej.sum(dim=0) / Ej[:, 0].count_nonzero(),
                dim=0
                )
            print(simj.item())
            sims.append(simj)
        sims, indices = torch.sort(torch.tensor(sims), descending=True)

        # TODO ignore anchor, E+i = fs(Si)\Ei
        # E_sorted = torch.empty((len(indices)-1, Ei.shape[0], Ei.shape[1]))
        # for idx in indices[1:]:
        #     print(idx.item())
        #     E_sorted[idx, :] = 
        #     print(simj.item())
        return E[indices]

    def cur_lev(self, E):
        """Curriculum level - Divide EEG signals into easy, medium and hard."""
        length = len(E)
        step = length // 3
        E_easy = E[:step]
        E_medium = E[step:2*step]
        E_hard = E[2*step:]
        return [E_easy, E_medium, E_hard]

    def cur_sche(self, curriculums, curr_level):
        """Curriculum scheduler - Sample a signal based on the level"""
        E_select = curriculums[curr_level]
        selected_E = np.random.choice(E_select)
        return selected_E


if __name__ == "__main__":

    # load data
    dataset_path_task1 = os.path.join(
        '../', 'dataset', 'ZuCo',
        'task1-SR', 'pickle', 'task1-SR-dataset.pickle'
        )

    with open(dataset_path_task1, 'rb') as handle:
        dataset_dict = [pickle.load(handle)]

    dataset = ZuCo(
        dataset_dict,
        'train',
        BartTokenizer.from_pretrained('facebook/bart-large'),
        subject='ALL',
        eeg_type='GD',
        bands='ALL',
        setting='unique_sent'
        )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # sample batch of eeg-text pairs (for a given subject)
    data_sample = next(iter(dataloader))
    EGG = data_sample[0]
    subject = data_sample[-2]
    sentence = data_sample[-1]

    # sample contrastive triplet
    fs, fp, S = build_CSCL_maps(dataset)
    cscl = CSCL(fs, fp, S)

    for level in range(3):
        triplet = cscl.get_triplet(EGG, subject, sentence, level)











