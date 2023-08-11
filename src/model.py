"""The model for EEG-to-Text decoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BartConfig, BartForConditionalGeneration


class BrainTranslator(nn.Module):
    """BrainTranslator model for EEG-to-Text decoding.

    Args:
    ----
        pretrained_seq2seq (nn.Module): Pretrained sequence-to-sequence model
        input_dim (int): Dimension of input eeg (105*num_bands) (default: 840)
        num_layers (int): Number of layers in the pre-encoder (default: 6)
        nhead (int): Number of heads in multiheadattention (default: 8)
        dim_pre_encoder (int): Pre-encoder hidden dimension (default: 2048)
        dim_s2s (int): The seq2seq model hidden dimension (default: 1024)

    """

    def __init__(self,
                 pretrained_seq2seq,
                 input_dim=840,
                 num_layers=6,
                 nhead=8,
                 dim_pre_encoder=2048,
                 dim_s2s=1024):

        super(BrainTranslator, self).__init__()

        self.pre_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_pre_encoder,
            batch_first=True
            )

        self.pre_encoder = nn.TransformerEncoder(
            self.pre_encoder_layer,
            num_layers=num_layers
            )

        self.fc = nn.Linear(
            in_features=input_dim, out_features=dim_s2s, bias=True
            )

        self.seq2seq = pretrained_seq2seq

    def forward(self, src, input_masks_batch, input_masks_invert, labels):
        """Forward pass of the BrainTranslator model.

        Args:
        ----
            src (Tensor): Word-level EEG (batch_size, seq_len, input_dim)
            input_masks_batch (Tensor): Input masks (0 is masked, 1 is not)
            input_masks_invert (Tensor): ~ Input masks (1 is masked, 0 is not)
            labels: (Tensor): Target labels

        Returns
        -------
            out (Tensor): The decoded EEG.

        """
        out = self.pre_encoder(src, src_key_padding_mask=input_masks_invert)

        out = F.relu(self.fc(out))

        out = self.seq2seq(
            inputs_embeds=out,
            attention_mask=input_masks_batch,
            return_dict=True,
            labels=labels
            )

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

    config = BartConfig.from_pretrained('facebook/bart-large')
    s2s = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    model = BrainTranslator(
        s2s,
        input_dim=input_dim,
        num_layers=num_layers,
        nhead=nhead,
        dim_pre_encoder=dim_pre_encoder,
        dim_s2s=dim_s2s
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # dummy sample for testing
    batch_size = 2
    seq_len = 56
    src = torch.rand(batch_size, seq_len, input_dim)
    input_masks_batch = torch.ones(batch_size, seq_len, dtype=torch.float)
    input_masks_invert = torch.zeros(batch_size, seq_len, dtype=torch.float)
    labels = torch.ones(batch_size, seq_len, dtype=torch.long)

    # forward pass and check output shape
    out = model(src, input_masks_batch, input_masks_invert, labels)
    assert out.logits.shape == torch.Size(
        [batch_size, seq_len, config.vocab_size]
        )


if __name__ == "__main__":
    main()
