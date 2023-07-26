"""Generate a tokenized sentence and EEG features for decoding.

Adapted from https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/data.py
"""

import numpy as np

import torch


def get_input_sample(sent_obj, tokenizer, eeg_type, bands, max_len=56):
    """Get a sample for a given sentence and subject EEG data.

    Args
    -------
        sent_obj (dict): A sentence object with EEG data.
        tokenizer: An instance of the tokenizer used to convert text to tokens.
        eeg_type (str): The type of eye-tracking features.
        bands (list): The EEG frequency bands to use.
        max_len (int, optional): Maximum length of the input. Defaults to 56.

    Returns
    -------
        input_sample (dict or None):
            - 'target_ids': Tokenized and encoded target sentence.
            - 'input_embeddings': Word-level EEG embeddings of the sentence.
            - 'input_attn_mask': Attention mask for input embeddings.
            - 'input_attn_mask_invert': Inverted attention mask.
            - 'target_mask': Attention mask for target sentence.
            - 'seq_len': Number of non-padding tokens in the sentence.

            Returns None if the input sentence is invalid or contains NaNs.
    """
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
