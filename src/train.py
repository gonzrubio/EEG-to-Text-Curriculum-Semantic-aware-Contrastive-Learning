"""Train an EEG-to-Text decoding model with Curriculum Semantic-aware Contrastive Learning."""

import copy
import os
import pickle
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

from CSCL import CSCL
from dataset import ZuCo, build_CSCL_maps
from model import BrainTranslatorPreEncoder, BrainTranslator
from utils.set_seed import set_seed


def train_BrainTranslator(
        model, dataloaders, loss_fn, optimizer, epochs, device
        ):
    pass


def train_CSCL(
        model, dataloaders, cscl, loss_fn, optimizer, epochs, device
        ):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    for level in range(1):
        best_loss = 100000000000

        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'dev']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                for EEG, _, _, _, _, _, subject, sentence in dataloaders[phase]:
                    # online sample of contrastive triplets
                    E, E_pos, E_neg = cscl.get_triplet(
                        EEG, subject, sentence, level
                        )
                    E = E.to(device)
                    E_pos = E_pos.to(device)
                    E_neg = E_neg.to(device)

                    # forward
                    # with torch.set_grad_enabled(phase == 'train'):
                        # seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)
                        # compute averaged vector of the outputs of the pre-encoder
                        # compute loss (equation 2)

                        # backward + optimize only if in training phase
                        # if phase == 'train':
                        #     # with torch.autograd.detect_anomaly():
                        #     optimizer.zero_grad(set to None)
                        #     loss.backward()
                        #     optimizer.step()

                    # statistics
                    # running_loss += loss.item() * input_embeddings_batch.size()[0] # batch loss

                # epoch_loss = running_loss / dataset_sizes[phase]
                # print(f'{phase} Loss: {epoch_loss:.4f}')

                # deep copy the model
                # if phase == 'dev' and epoch_loss < best_loss:
                #     best_loss = epoch_loss
                #     best_model_wts = copy.deepcopy(model.state_dict())

            # print()

    # time_elapsed = time.time() - since
    # print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # print(f'Best val loss: {best_loss:4f}')

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


def main():
    cfg = {
        'seed': 312,
        'subject_choice': 'ALL',
        'eeg_type_choice': 'GD',
        'bands_choice': 'ALL',
        'dataset_setting': 'unique_sent',
        'batch_size': 1,  # 32
        'shuffle': False,
        'input_dim': 840,
        'num_layers': 6,
        'nhead': 8,
        'dim_pre_encoder': 2048,
        'dim_s2s': 1024,
        'temp': 1e-5,
        'lr_pre': 1e-3,
        'epochs_pre': 1,
        'lr': 2e-5,
        'epochs': 1
        }

    set_seed(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data and dataloaders
    whole_dataset_dicts = []

    dataset_path_task1 = os.path.join(
        '../', 'dataset', 'ZuCo',
        'task1-SR', 'pickle', 'task1-SR-dataset.pickle'
        )

    whole_dataset_dicts = []
    for t in [dataset_path_task1]:
        with open(t, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    train_set = ZuCo(
        whole_dataset_dicts,
        'train',
        BartTokenizer.from_pretrained('facebook/bart-large'),
        subject=cfg['subject_choice'],
        eeg_type=cfg['eeg_type_choice'],
        bands=cfg['bands_choice'],
        setting=cfg['dataset_setting']
        )

    train_loader = DataLoader(
        train_set, batch_size=cfg['batch_size'], shuffle=cfg['shuffle']
        )

    val_set = ZuCo(
        whole_dataset_dicts,
        'dev',
        BartTokenizer.from_pretrained('facebook/bart-large'),
        subject=cfg['subject_choice'],
        eeg_type=cfg['eeg_type_choice'],
        bands=cfg['bands_choice'],
        setting=cfg['dataset_setting']
        )

    val_loader = DataLoader(
        val_set, batch_size=cfg['batch_size'], shuffle=cfg['shuffle']
        )

    dataloaders = {'train': train_loader, 'val': val_loader}

    # train pre-encoder with CSCL
    model = BrainTranslatorPreEncoder(
        input_dim=cfg['input_dim'],
        num_layers=cfg['num_layers'],
        nhead=cfg['nhead'],
        dim_pre_encoder=cfg['dim_pre_encoder'],
        dim_s2s=cfg['dim_s2s']
        ).to(device)

    fs, fp, S = build_CSCL_maps(train_set)
    cscl = CSCL(fs, fp, S)
    loss_fn = cfg['temp']  # TODO
    optimizer = optim.Adam(params=model.parameters(), lr=cfg['lr_pre'])

    model = train_CSCL(
        model, dataloaders, cscl, loss_fn, optimizer, cfg['epochs_pre'], device
        )

    # train BrainTranslator
    model = BrainTranslator(
        model,
        BartForConditionalGeneration.from_pretrained('facebook/bart-large'),
    ).to(device)

    loss_fn = None
    optimizer = optim.Adam(params=model.parameters(), lr=cfg['lr'])

    model = train_BrainTranslator(
        model, dataloaders, loss_fn, optimizer, cfg['epochs'], device
        )


if __name__ == "__main__":
    main()
