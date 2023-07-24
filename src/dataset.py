"""ML-ready ZuCo dataset for EEG-to-Text decoding.

Adapted from https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/data.py
"""

import os

import pickle
from tqdm import tqdm

import numpy as np

import torch
from transformers import BartTokenizer
from torch.utils.data import Dataset


def get_input_sample(key, i, sent_obj, tokenizer, eeg_type, bands, max_len=56):
    """From https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/data.py."""

    def normalize_1d(input_tensor):
        mean = torch.mean(input_tensor)
        std = torch.std(input_tensor)
        input_tensor = (input_tensor - mean)/std
        return input_tensor

    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands:
            frequency_features.append(
                word_obj['word_level_EEG'][eeg_type][eeg_type+band]
                )
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105*len(bands):
            # print(f'expect word eeg embedding dim to be {105*len(bands)},
            # but got {len(word_eeg_embedding)}, return None')
            return None
        # assert len(word_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)

    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []
        for band in bands:
            key = 'mean'+band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    if sent_obj is None:
        # print(f'  - skip bad sentence')
        return None

    input_sample = {}

    # get target label
    target_string = sent_obj['content']
    target_tokenized = tokenizer(
        target_string, padding='max_length', max_length=max_len,
        truncation=True, return_tensors='pt', return_attention_mask=True
        )
    input_sample['target_ids'] = target_tokenized['input_ids'][0]

    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        # print('[NaN sent level eeg]: ', target_string)
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # handle some wierd case
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty', 'empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1', 'film.')

    # get input embeddings
    word_embeddings = []
    for word in sent_obj['word']:
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(
            word, eeg_type, bands=bands
            )
        if word_level_eeg_tensor is None:   # check none, for v2 dataset
            return None
        if torch.isnan(word_level_eeg_tensor).any():
            # print('[NaN ERROR] problem sent:',sent_obj['content'])
            # print('[NaN ERROR] problem word:',word['content'])
            # print('[NaN ERROR] problem word feature:',word_level_eeg_tensor)
            return None
        word_embeddings.append(word_level_eeg_tensor)

    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))

    # input_sample['input_embeddings'].shape = max_len * (105*num_bands)
    input_sample['input_embeddings'] = torch.stack(word_embeddings)
    len_sent_word = len(sent_obj['word'])  # len_sent_word <= max_len

    # mask out padding tokens, 0 is masked out, 1 is not masked
    input_sample['input_attn_mask'] = torch.zeros(max_len)
    input_sample['input_attn_mask'][:len_sent_word] = torch.ones(len_sent_word)

    # mask out padding tokens reverted: handle different use case: this is for
    # pytorch transformers. 1 is masked out, 0 is not masked
    input_sample['input_attn_mask_invert'] = torch.ones(max_len)
    input_sample['input_attn_mask_invert'][:len_sent_word] = torch.zeros(
        len_sent_word
        )

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])

    # clean 0 length data
    if input_sample['seq_len'] == 0:
        # print('discard length zero instance: ', target_string)
        return None

    return input_sample


class ZuCo(Dataset):
    # dosctring:
    # What does it do?
    # A: convert pickle files for each task and all subjects into a...
    # describe splits, constructor arguments, input tensor size: torch.Size([56, 840])
    # take first 80% as trainset, 10% as dev and 10% as test
    # TODO this info bellow should go in class dosctring explaining
    # the difference between unique subject and unique sentence
    # print('WARNING!!! only implemented for SR v1 dataset ')
    # subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW']
    # for train
    # subject ['ZMG'] for dev
    # subject ['ZPH'] for test

    def __init__(self,
                 input_dataset_dicts,
                 phase,
                 tokenizer,
                 subject='ALL',
                 eeg_type='GD',
                 bands='ALL',
                 setting='unique_sent'):

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]

        self.inputs = []
        self.tokenizer = tokenizer
        self.subject = subject
        self.setting = setting
        self.eeg_type = eeg_type
        self.train = 0.8
        self.dev = 0.1
        self.bands = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'] \
            if bands == 'ALL' else bands

        # go through all task datasets
        for input_dataset_dict in input_dataset_dicts:

            # get the subject(s) key/name for this task
            subjects = list(input_dataset_dict.keys()) \
                if subject == 'ALL' else [subject]

            # number of sentences per subject in this task
            total_num_sentence = len(input_dataset_dict[subjects[0]])

            # create dataset grouped by unique sentence or subject
            if setting == 'unique_sent':
                self.unique_sent(
                    phase, subjects, input_dataset_dict, total_num_sentence
                    )
            elif setting == 'unique_subj':
                self.unique_subj(phase, input_dataset_dict, total_num_sentence)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'],
            input_sample['seq_len'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'],
            input_sample['target_mask']
        )

    def __len__(self):
        return len(self.inputs)

    def unique_sent(
            self, phase, subjects, input_dataset_dict, total_num_sentence
            ):
        # indices separating the sentences into train/dev/test splits
        train_divider = int(self.train * total_num_sentence)
        dev_divider = train_divider + int(self.dev * total_num_sentence)

        if phase == 'train':
            range_iter = range(train_divider)
        elif phase == 'dev':
            range_iter = range(train_divider, dev_divider)
        elif phase == 'test':
            range_iter = range(dev_divider, total_num_sentence)

        for key in subjects:
            for i in range_iter:
                self.append_input_sample(input_dataset_dict, key, i)

    def unique_subj(
            self, phase, input_dataset_dict, total_num_sentence
            ):
        # sort the subjetcs into train/dev/test splits
        if phase == 'train':
            subj_iter = [
                'ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'
                ]
        elif phase == 'dev':
            subj_iter = ['ZMG']
        elif phase == 'test':
            subj_iter = ['ZPH']

        for i in range(total_num_sentence):
            for key in subj_iter:
                self.append_input_sample(input_dataset_dict, key, i)

    def append_input_sample(self, input_dataset_dict, key, i):
        input_sample = get_input_sample(
            key, i,
            input_dataset_dict[key][i],
            self.tokenizer,
            self.eeg_type,
            self.bands
            )
        if input_sample is not None:
            self.inputs.append(input_sample)


def main():
    """ML-ready ZuCo dataset sanity check."""
    # load the pickle files for all tasks
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

    # check the number of subjects and unique sentences in each task
    for idx, dataset_dict in enumerate(whole_dataset_dicts):
        if idx == 0:
            num_sent = 400
            num_subj = 12
        elif idx == 1:
            num_sent = 300
            num_subj = 12
        else:
            num_sent = 349
            num_subj = 18

        assert len(dataset_dict) == num_subj

        for key in dataset_dict:
            assert len(dataset_dict[key]) == num_sent

    # data config
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    subject_choice = 'ALL'
    eeg_type_choice = 'GD'
    bands_choice = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
    dataset_setting = 'unique_sent'

    # check split size
    for split in tqdm(['train', 'dev', 'test']):
        dataset = ZuCo(
            whole_dataset_dicts,
            split,
            tokenizer,
            subject=subject_choice,
            eeg_type=eeg_type_choice,
            bands=bands_choice,
            setting=dataset_setting
            )
        print(f'{split}set size:', len(dataset))


if __name__ == '__main__':
    main()
