"""ML-ready ZuCo dataset for EEG-to-Text decoding.

Adapted from https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/data.py
"""

import os
import pickle
from tqdm import tqdm

from transformers import BartTokenizer
from torch.utils.data import Dataset

from data.get_input_sample import get_input_sample


class ZuCo(Dataset):
    # dosctring:
    # What does it do? ie explain constructor and getter method
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
    bands_choice = 'ALL'
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
        print(f' {split}set size:', len(dataset))


if __name__ == '__main__':
    main()
