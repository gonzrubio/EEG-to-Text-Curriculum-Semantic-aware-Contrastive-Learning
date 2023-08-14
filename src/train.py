"""Train an EEG-to-Text decoding model with Curriculum Semantic-aware Contrastive Learning."""

import os
import pickle

import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

from dataset import ZuCo
from model import BrainTranslator
from utils.set_seed import set_seed


def train_CSCL():
    pass


def main():
    # configuration
    # TODO refactor in utils?
    cfg = {
        'seed': 312,
        'subject_choice': 'ALL',
        'eeg_type_choice': 'GD',
        'bands_choice': 'ALL',
        'dataset_setting': 'unique_sent',
        'batch_size': 1,
        'shuffle': False,
        'input_dim': 840,
        'num_layers': 6,
        'nhead': 8,
        'dim_pre_encoder': 2048,
        'dim_s2s': 1024,
        }

    set_seed(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up the dataloader
    whole_dataset_dicts = []

    dataset_path_task1 = os.path.join(
        '../', 'dataset', 'ZuCo',
        'task1-SR', 'pickle', 'task1-SR-dataset.pickle'
        )

    whole_dataset_dicts = []
    for t in [dataset_path_task1]:
        with open(t, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    # data config
    train_set = ZuCo(
        whole_dataset_dicts,
        'train',
        BartTokenizer.from_pretrained('facebook/bart-large'),
        subject=cfg['subject_choice'],
        eeg_type=cfg['eeg_type_choice'],
        bands=cfg['bands_choice'],
        setting=cfg['dataset_setting']
        )

    train_dataloader = DataLoader(
        train_set, batch_size=cfg['batch_size'], shuffle=cfg['shuffle']
        )

    # model
    model = BrainTranslator(
        BartForConditionalGeneration.from_pretrained('facebook/bart-large'),
        input_dim=cfg['input_dim'],
        num_layers=cfg['num_layers'],
        nhead=cfg['nhead'],
        dim_pre_encoder=cfg['dim_pre_encoder'],
        dim_s2s=cfg['dim_s2s']
    ).to(device)

    # train pre-encoder with CSCL
    # return a batch of samples from loader
    # for each sample in the batch construct a triplet
    # compute  on batch (equation 2)

    # verify loss @ init -log(1/n_classes)

    # input-indepent baseline.

    # overfit small(est) batch


if __name__ == "__main__":
    main()
