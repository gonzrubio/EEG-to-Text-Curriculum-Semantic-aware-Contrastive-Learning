"""The model for EEG-to-Text decoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BartConfig, BartForConditionalGeneration


class BrainTranslator(nn.Module):
    """BrainTranslator model for EEG-to-Text decoding.

    Args:
    ----
        pre_encoder: Pre-encoder trained with CSCL
        pretrained_seq2seq: Pretrained sequence-to-sequence model

    """

    def __init__(self, pre_encoder, pretrained_seq2seq):
        super(BrainTranslator, self).__init__()
        self.pre_encoder = pre_encoder
        self.seq2seq = pretrained_seq2seq

    def forward(self, src, mask_pre_encoder, mask_seq2seq, labels):
        """Forward pass of the BrainTranslator model.

        Args:
        ----
            src (Tensor): Word-level EEG (batch_size, seq_len, input_dim)
            mask_pre_encoder (Tensor): Input masks (1 is masked, 0 is not)
            mask_seq2seq (Tensor): ~ Input masks (0 is masked, 1 is not)
            labels: (Tensor): Target labels

        Returns
        -------
            out (Tensor): The decoded EEG.

        """
        out = self.pre_encoder(src, mask_pre_encoder)
        out = self.seq2seq(
            inputs_embeds=out,
            attention_mask=mask_seq2seq,
            return_dict=True,
            labels=labels
            )
        return out


class BrainTranslatorPreEncoder(nn.Module):
    """Pre-encoder module for BrainTranslator.

    Args:
    ----
        input_dim (int): Dimension of input eeg (105*num_bands) (default: 840)
        num_layers (int): Number of layers in the pre-encoder (default: 6)
        nhead (int): Number of heads in multiheadattention (default: 8)
        dim_pre_encoder (int): Pre-encoder hidden dimension (default: 2048)
        dim_s2s (int): The seq2seq model hidden dimension (default: 1024)

    """

    def __init__(self,
                 input_dim=840,
                 num_layers=6,
                 nhead=8,
                 dim_pre_encoder=2048,
                 dim_s2s=1024,
                 dropout=0.1):

        super(BrainTranslatorPreEncoder, self).__init__()

        self.pre_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_pre_encoder,
            dropout=dropout,
            batch_first=True,
            norm_first=True
            )
        self.pre_encoder_transformer = nn.TransformerEncoder(
            self.pre_encoder_layer,
            num_layers=num_layers
            )
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=input_dim, out_features=dim_s2s, bias=True)
            )
        # self.fc = nn.Linear(
        #     in_features=input_dim, out_features=dim_s2s, bias=True
        #     )
        # self.fc2 = nn.Linear(
        #     in_features=dim_s2s, out_features=dim_s2s, bias=True
        #     )

    def forward(self, src, mask_pre_encoder):
        """Forward pass of the BrainTranslatorPreEncoder model.

        Args:
        ----
            src (Tensor): Word-level EEG (batch_size, seq_len, input_dim)
            mask_pre_encoder (Tensor): Input masks (1 is masked, 0 is not)

        Returns
        -------
            out (Tensor): The decoded EEG.

        """
        out = self.pre_encoder_transformer(
            src, src_key_padding_mask=mask_pre_encoder
            )
        # out = F.relu(self.fc(out))
        # out = self.fc2(out)
        out = self.fc(out)
        return out


def main():
    """Example usage."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    input_dim = 840         # dimension of input eeg feature
    num_layers = 6          # pre-encoder number of layers
    nhead = 8               # number of heads in each pre-encoder layer
    dim_pre_encoder = 2048  # hidden dim for pre encoder
    dim_s2s = 1024          # hidden dim for seq2seq model

    pre_encoder = BrainTranslatorPreEncoder(
        input_dim=input_dim,
        num_layers=num_layers,
        nhead=nhead,
        dim_pre_encoder=dim_pre_encoder,
        dim_s2s=dim_s2s
        )

    num_params = sum(p.numel() for p in pre_encoder.parameters() if p.requires_grad)
    print(f"{num_params:,} trainable parameters in the pre-encoder")

    config = BartConfig.from_pretrained('facebook/bart-large')
    s2s = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    model = BrainTranslator(pre_encoder, s2s).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{num_params:,}trainable parameters in BrainTranslator")

    # dummy sample for testing
    batch_size = 64
    seq_len = 56
    src = torch.rand(batch_size, seq_len, input_dim).to(device)
    input_masks_batch = torch.ones(batch_size, seq_len, dtype=torch.float).to(device)
    input_masks_invert = torch.zeros(batch_size, seq_len, dtype=torch.float).to(device)
    labels = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)

    # forward pass and check output shape
    out = model(src, input_masks_batch, input_masks_invert, labels)
    assert out.logits.shape == torch.Size(
        [batch_size, seq_len, config.vocab_size]
        )


if __name__ == "__main__":
    main()
