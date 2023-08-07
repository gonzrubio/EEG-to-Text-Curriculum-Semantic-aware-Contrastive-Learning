"""The model for EEG-to-Text decoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# chatgpt
# class BrainTranslator(nn.Module):
#     def __init__(self, word_input_dim, word_output_dim, pre_encoder_input_dim, pre_encoder_hidden_dim, pre_encoder_output_dim, seq2seq_input_dim, seq2seq_hidden_dim, seq2seq_output_dim):
#         super(BrainTranslator, self).__init__()
#         self.word_feature_constructor = self.WordEEGFeatureConstruction(word_input_dim, word_output_dim)
#         self.pre_encoder = self.PreEncoder(pre_encoder_input_dim, pre_encoder_hidden_dim, pre_encoder_output_dim)
#         self.seq2seq = self.Seq2Seq(seq2seq_input_dim, seq2seq_hidden_dim, seq2seq_output_dim)

#     def forward(self, input_features):
#         word_eeg_features = self.word_feature_constructor(input_features)
#         pre_encoder_output = self.pre_encoder(word_eeg_features)
#         output_sentence = self.seq2seq(pre_encoder_output)
#         return output_sentence

#     def WordEEGFeatureConstruction():
#         pass

#     def PreEncoder():
#         pass

#     def Seq2Seq():
#         pass


# cerelyze
# class BrainTranslator(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super(BrainTranslator, self).__init__()

#         # Pre-encoder layers
#         self.pre_encoder = nn.TransformerEncoderLayer(input_dim, hidden_dim, num_layers)

#         # Pre-trained seq2seq model layers
#         self.decoder = nn.TransformerDecoderLayer(hidden_dim, hidden_dim, num_layers)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.softmax = nn.Softmax(dim=2)

#     def forward(self, input_data):
#         # Pre-encoder
#         pre_encoder_output = self.pre_encoder(input_data)

#         # Pre-trained seq2seq model
#         decoder_output = self.decoder(pre_encoder_output)
#         fc_output = self.fc(decoder_output)
#         softmax_output = self.softmax(fc_output)

#         return softmax_output


class BrainTranslator(nn.Module):
    """BrainTranslator model for EEG-to-Text decoding.

    Args:
    ----
        pretrained_seq2seq (nn.Module): Pretrained sequence-to-sequence model.
        d_model (int): Input and output embeddings dimension (default: 840).
        nhead (int): Number of heads in multiheadattention (default: 8).
        dim_decoder (int): Decoder's embedding layer dimension (default: 1024).
        dim_feedforward (int): Fully connected net dimension (default: 2048).
        num_layers (int): Number of transformer encoder layers (default: 6).

    """

    def __init__(self,
                 pretrained_seq2seq,
                 d_model=840,
                 nhead=8,
                 dim_feedforward=2048,
                 dim_decoder=1024,
                 num_layers=6):

        super(BrainTranslator, self).__init__()

        self.pre_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
            )

        self.pre_encoder = nn.TransformerEncoder(
            self.pre_encoder_layer,
            num_layers=num_layers
            )

        self.fc = nn.Linear(
            in_features=d_model, out_features=dim_decoder, bias=True
            )

        self.seq2seq = pretrained_seq2seq

    def forward(self, src, input_masks_batch, input_masks_invert, labels):
        """
        Note:
        ----
            - `input_masks_batch` and `input_masks_invert` are used for masking during attention computations.
            - The `pretrained_seq2seq` module should be compatible with the desired sequence-to-sequence task.

        input_embeddings_batch: batch_size*Seq_len*840
        input_mask: 1 is not masked, 0 is masked
        input_masks_invert: 1 is masked, 0 is not masked

        Parameters
        ----------
        src : TYPE
            DESCRIPTION.
        input_masks_batch : TYPE
            DESCRIPTION.
        input_masks_invert : TYPE
            DESCRIPTION.
        target_ids_batch_converted : TYPE
            DESCRIPTION.

        Returns
        -------
        out : TYPE
            DESCRIPTION.

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
    """BrainBART example usage."""
    # Define model parameters
    input_dim = 128
    hidden_dim = 256
    output_dim = 10
    num_layers = 6

    # Create an instance of the BrainTranslator model
    model = BrainTranslator(input_dim, hidden_dim, output_dim, num_layers)

    # Generate fake input data for testing
    # shape: (batch_size, sequence_length, input_dim)
    batch_size = 32
    sequence_length = 10
    input_data = torch.randn(
        batch_size, sequence_length, input_dim
        )

    # Pass the input data through the model
    # shape: (batch_size, sequence_length, output_dim)
    output = model(input_data)
    print(output.shape)


if __name__ == "__main__":
    main()
