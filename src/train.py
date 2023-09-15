"""Train an EEG-to-Text decoding model with Curriculum Semantic-aware Contrastive Learning."""

import copy
import os
import pickle
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
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
        model, dataloaders, cscl, loss_fn, optimizer, epochs, device, wnb
        ):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    for level in range(3):
        best_loss = 100000000000

        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs - 1}')
            print('-' * 10)

            for phase in ['dev', 'train']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                loader = dataloaders[phase]

                for batch, (EEG, _, _, _, _, _, subject, sentence) in enumerate(loader):

                    E, E_pos, E_neg, mask, mask_pos, mask_neg = cscl[phase].get_triplet(
                        EEG, subject, sentence, level
                        )

                    with torch.set_grad_enabled(phase == 'train'):
                        mask_triplet = torch.vstack((mask, mask_pos, mask_neg)).to(device)
                        out = model(
                            torch.vstack((E, E_pos, E_neg)).to(device),
                            mask_triplet
                            )
                        # out = torch.mean(out, dim=1)
                        # invert mask and average pre-encoder outputs
                        mask_triplet = abs(mask_triplet-1).unsqueeze(-1)
                        out = (out * mask_triplet).sum(1) / mask_triplet.sum(1)

                        h = out[:E.size(0), :]
                        h_pos = out[E.size(0):2*E.size(0), :]
                        h_neg = out[2*E.size(0):, :]
                        # h = torch.mean(out, dim=1)
                        # h = h.view(-1, 3, h.shape[-1])

                        T = 1
                        num = torch.exp(F.cosine_similarity(h, h_pos, dim=1)/T)
                        denom = torch.empty_like(num, device=num.device)
                        for j in range(E.size(0)):
                            denomjj = 0
                            for jj in range(E.size(0)):
                                denomjj += torch.exp(F.cosine_similarity(h[j, :], h_pos[jj, :], dim=0)/T)
                                denomjj += torch.exp(F.cosine_similarity(h[j, :], h_neg[jj, :], dim=0)/T)
                            denom[j] = denomjj

                        # num = torch.exp(
                        #     F.cosine_similarity(h[:, 0, :], h[:, 1, :], dim=1) / T
                        #     )
                        # denom = torch.empty_like(num, device=num.device)
                        # for j in range(E.size(0)):
                        #     denomjj = 0
                        #     for jj in range(E.size(0)):
                        #         denomjj += torch.exp(F.cosine_similarity(h[j, 0, :], h[jj, 1, :], dim=0) / T)
                        #         denomjj += torch.exp(F.cosine_similarity(h[j, 0, :], h[jj, 2, :], dim=0) / T)
                        #     denom[j] = denomjj

                        loss = -torch.log(num / denom).mean()
                        print(f'{epoch}.{batch} {phase} Loss: {loss:.4f}')
                        # print(f'{epoch}.{batch} {phase} Loss: {loss:.4e}')

                        if phase == 'train':
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()

                        if wnb:
                            wandb.log({f"{phase} batch loss": loss.item()})

                        running_loss += loss.item()

                epoch_loss = running_loss / len(loader)
                print(f'{phase} Loss: {epoch_loss:.4f}')
                # print(f'{phase} Loss: {epoch_loss:.4e}')

                if wnb:
                    wandb.log({f"{phase} epoch loss": epoch_loss})

                if phase == 'dev' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:4f}')

    model.load_state_dict(best_model_wts)
    return model


def main():
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
        'T': 5e-6,
        'lr_pre': 1e-6,
        'epochs_pre': 5,
        'lr': 1e-6,
        'epochs': 5,
        'wandb': True
        }

    if cfg['wandb']:
        wandb.init(project='CSCL', reinit=True, config=cfg)

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

    dev_set = ZuCo(
        whole_dataset_dicts,
        'dev',
        BartTokenizer.from_pretrained('facebook/bart-large'),
        subject=cfg['subject_choice'],
        eeg_type=cfg['eeg_type_choice'],
        bands=cfg['bands_choice'],
        setting=cfg['dataset_setting']
        )

    dev_loader = DataLoader(
        dev_set, batch_size=cfg['batch_size'], shuffle=cfg['shuffle']
        )

    dataloaders = {'train': train_loader, 'dev': dev_loader}

    # train pre-encoder with CSCL
    model = BrainTranslatorPreEncoder(
        input_dim=cfg['input_dim'],
        num_layers=cfg['num_layers'],
        nhead=cfg['nhead'],
        dim_pre_encoder=cfg['dim_pre_encoder'],
        dim_s2s=cfg['dim_s2s']
        ).to(device)

    if cfg['wandb']:
        wandb.watch(model, log='all')

    fs, fp, S = build_CSCL_maps(train_set)
    cscl_train = CSCL(fs, fp, S)

    fs, fp, S = build_CSCL_maps(dev_set)
    cscl_dev = CSCL(fs, fp, S)

    cscl = {'train': cscl_train, 'dev': cscl_dev}

    loss_fn = cfg['T']  # TODO
    optimizer = optim.Adam(params=model.parameters(), lr=cfg['lr_pre'])

    model = train_CSCL(
        model, dataloaders, cscl, loss_fn, optimizer, cfg['epochs_pre'], device, cfg['wandb']
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
