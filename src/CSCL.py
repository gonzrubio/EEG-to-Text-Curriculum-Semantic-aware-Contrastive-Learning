"""Curriculum Semantic-aware Contrastive Learning Framework."""

import os
import pickle
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from utils.HashTensor import HashTensor
from dataset import ZuCo, build_CSCL_maps


class CSCL:
    """Curriculum Semantic-aware Contrastive Learning (CSCL).

    Generate contrastive triplets given the EEG signals generated by the
    subjects for a given sentence, and curriculum level.

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
        E_positive_curriculum = torch.empty_like(Ei)
        E_negative_curriculum = torch.empty_like(Ei)

        # prepare batch of contrastive triplets ##### also zip eeg
        for i, (eeg, s, p) in enumerate(zip(Ei, Si, pi)):
            eeg = HashTensor(eeg)

            # Positive pairs
            E_positive = set(self.fs[s])
            E_positive.remove(eeg)
            E_positive_sorted = self.cur_cri(eeg, E_positive, descending=True)
            curriculums = self.cur_lev(E_positive_sorted)
            E_positive_curriculum[i] = self.cur_sche(curriculums, curr_level)

            # Negative pairs
            E_negative = set(self.S)
            E_negative.remove(s)
            E_negative = {e for ss in E_negative for e in self.fs.get(ss, set())}
            E_negative.difference_update(self.fp[p])
            E_negative_sorted = self.cur_cri(eeg, E_negative, descending=False)
            curriculums = self.cur_lev(E_negative_sorted)
            E_negative_curriculum[i] = self.cur_sche(curriculums, curr_level)

        return Ei, E_positive_curriculum, E_negative_curriculum

    def cur_cri(self, Ei, E, descending):
        """Curriculum criterion - sort the EEG signals based on similarity."""
        sims = []
        E_sorted = []
        for Ej in E:
            simj = F.cosine_similarity(
                Ei.sum(dim=0) / Ei[:, 0].count_nonzero(),
                Ej.sum(dim=0) / Ej[:, 0].count_nonzero(),
                dim=0
                )
            sims.append(simj)
            E_sorted.append(Ej)
        sims, indices = torch.sort(torch.tensor(sims), descending=descending)
        E_sorted = [E_sorted[j].unsqueeze(0) for j in indices]
        E_sorted = torch.cat(E_sorted, dim=0)
        return E_sorted

    def cur_lev(self, E):
        """Curriculum level - Divide EEG signals into easy, medium and hard."""
        length = len(E)
        step = length // 3
        E_easy = E[:step]
        E_medium = E[step:2*step]
        E_hard = E[2*step:]
        return [E_easy, E_medium, E_hard]

    def cur_sche(self, curriculums, curr_level):
        """Curriculum scheduler - Sample a signal based on the level."""
        E_select = curriculums[curr_level]
        E = E_select[random.randint(0, E_select.shape[0]-1)]
        return E


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

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # sample batch of eeg-text pairs (for a given subject)
    data_sample = next(iter(dataloader))
    EEG = data_sample[0]
    sentence = data_sample[-1]
    subject = data_sample[-2]

    # sample contrastive triplet
    fs, fp, S = build_CSCL_maps(dataset)
    cscl = CSCL(fs, fp, S)

    for curr_level in range(3):
        E, E_pos, E_neg = cscl.get_triplet(EEG, subject, sentence, curr_level)
        assert E.shape == E_pos.shape == E_neg.shape == EEG.shape

        print(f'\ncurriculum level {curr_level}')
        print('-------------------')
        print('positive  negative')
        print('--------  --------')
        for e, e_pos, e_neg in zip(E, E_pos, E_neg):
            sim_pos = F.cosine_similarity(
                e.sum(dim=0) / E[:, 0].count_nonzero(),
                e_pos.sum(dim=0) / e_pos[:, 0].count_nonzero(),
                dim=0
                )

            sim_neg = F.cosine_similarity(
                e.sum(dim=0) / E[:, 0].count_nonzero(),
                e_neg.sum(dim=0) / e_neg[:, 0].count_nonzero(),
                dim=0
                )
            print(f'{sim_pos.item():.4f}    {sim_neg.item():.4f}')
