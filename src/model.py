"""The model for EEG-to-Text decoding."""

import torch
import torch.nn as nn


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
    def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslator, self).__init__()

        self.pretrained = pretrained_layers
        # additional transformer encoder, following BART paper about 
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)

        # print('[INFO]adding positional embedding')
        # self.positional_embedding = PositionalEncoding(in_feature)

        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""

        # input_embeddings_batch = self.positional_embedding(input_embeddings_batch) 

        # use src_key_padding_masks
        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask = input_masks_invert) 

        # encoded_embedding = self.additional_encoder(input_embeddings_batch) 
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch, return_dict = True, labels = target_ids_batch_converted)                    

        return out

def main():
    # Example usage

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
