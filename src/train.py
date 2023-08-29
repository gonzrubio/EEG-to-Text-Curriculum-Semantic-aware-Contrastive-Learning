"""Train an EEG-to-Text decoding model with Curriculum Semantic-aware Contrastive Learning."""

import os
import pickle

import torch
import torch.nn.functional as F
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
    pass
    # for level in range(1):
    #     for data in train_dataloader:
    #         EEG = data[0]
    #         sentence = data[-1]
    #         subject = data[-2]

    #         # online sample of contrastive triplets
    #         E, E_pos, E_neg = cscl.get_triplet(EEG, subject, sentence, level)
    #         # TODO move to device here

    #         print(f'\ncurriculum level {level}')
    #         print('-------------------')
    #         print('positive  negative')
    #         print('--------  --------')
    #         for e, e_pos, e_neg in zip(E, E_pos, E_neg):
    #             sim_pos = F.cosine_similarity(
    #                 e.sum(dim=0) / E[:, 0].count_nonzero(),
    #                 e_pos.sum(dim=0) / e_pos[:, 0].count_nonzero(),
    #                 dim=0
    #                 )

    #             sim_neg = F.cosine_similarity(
    #                 e.sum(dim=0) / E[:, 0].count_nonzero(),
    #                 e_neg.sum(dim=0) / e_neg[:, 0].count_nonzero(),
    #                 dim=0
    #                 )
    #             print(f'{sim_pos.item():.4f}    {sim_neg.item():.4f}')
    # fwd pass triplet through pre-encoder (flag in fwd for pre-encoder only)
    # compute averaged vector of the outputs of the pre-encoder
    # compute loss on batch (equation 2) (start with no temp, see how it varies with temp)
    # verify loss @ init -log(1/n_classes)
    # compute input-indepent baseline.
    # overfit small(est) batch
    # since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_loss = 100000000000

    # for epoch in range(epochs):
    #     print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    #     print('-' * 10)

    #     # Each epoch has a training and validation phase
    #     for phase in ['train', 'dev']:
    #         if phase == 'train':
    #             model.train()  # Set model to training mode
    #         else:
    #             model.eval()   # Set model to evaluate mode

    #         running_loss = 0.0

    #         # Iterate over data.
    #         for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(dataloaders[phase]):

    #             # load in batch
    #             input_embeddings_batch = input_embeddings.to(device).float()
    #             input_masks_batch = input_masks.to(device)
    #             input_mask_invert_batch = input_mask_invert.to(device)
    #             target_ids_batch = target_ids.to(device)
    #             """replace padding ids in target_ids with -100"""
    #             target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

    #             # zero the parameter gradients
    #             optimizer.zero_grad()

    #             # forward
    #             seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)

    #             """calculate loss"""
    #             # logits = seq2seqLMoutput.logits # 8*48*50265
    #             # logits = logits.permute(0,2,1) # 8*50265*48

    #             # loss = criterion(logits, target_ids_batch_label) # calculate cross entropy loss only on encoded target parts
    #             # NOTE: my criterion not used
    #             loss = seq2seqLMoutput.loss # use the BART language modeling loss

    #             # """check prediction, instance 0 of each batch"""
    #             # print('target size:', target_ids_batch.size(), ',original logits size:', logits.size(), ',target_mask size', target_mask_batch.size())
    #             # logits = logits.permute(0,2,1)
    #             # for idx in [0]:
    #             #     print(f'-- instance {idx} --')
    #             #     # print('permuted logits size:', logits.size())
    #             #     probs = logits[idx].softmax(dim = 1)
    #             #     # print('probs size:', probs.size())
    #             #     values, predictions = probs.topk(1)
    #             #     # print('predictions before squeeze:',predictions.size())
    #             #     predictions = torch.squeeze(predictions)
    #             #     # print('predictions:',predictions)
    #             #     # print('target mask:', target_mask_batch[idx])
    #             #     # print('[DEBUG]target tokens:',tokenizer.decode(target_ids_batch_copy[idx]))
    #             #     print('[DEBUG]predicted tokens:',tokenizer.decode(predictions))

    #             # backward + optimize only if in training phase
    #             if phase == 'train':
    #                 # with torch.autograd.detect_anomaly():
    #                 loss.backward()
    #                 optimizer.step()

    #             # statistics
    #             running_loss += loss.item() * input_embeddings_batch.size()[0] # batch loss
    #             # print('[DEBUG]loss:',loss.item())
    #             # print('#################################')

    #         if phase == 'train':
    #             scheduler.step()

    #         epoch_loss = running_loss / dataset_sizes[phase]

    #         print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    #         # deep copy the model
    #         if phase == 'dev' and epoch_loss < best_loss:
    #             best_loss = epoch_loss
    #             best_model_wts = copy.deepcopy(model.state_dict())
    #             '''save checkpoint'''
    #             torch.save(model.state_dict(), checkpoint_path_best)
    #             print(f'update best on dev checkpoint: {checkpoint_path_best}')
    #     print()

    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val loss: {:4f}'.format(best_loss))
    # torch.save(model.state_dict(), checkpoint_path_last)
    # print(f'update last checkpoint: {checkpoint_path_last}')

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    # return model


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
