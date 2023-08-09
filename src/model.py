"""The model for EEG-to-Text decoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.set_seed import set_seed


class BrainTranslator(nn.Module):
    """BrainTranslator model for EEG-to-Text decoding.

    Args:
    ----
        pretrained_seq2seq (nn.Module): Pretrained sequence-to-sequence model
        d_model (int): Input and output embeddings dimension (default: 840)
        nhead (int): Number of heads in multiheadattention (default: 8)
        dim_decoder (int): Decoder's embedding layer dimension (default: 1024)
        dim_feedforward (int): Fully connected net dimension (default: 2048)
        num_layers (int): Number of transformer encoder layers (default: 6)

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
        """Forward pass of the BrainTranslator model.

        Args:
        ----
            src (Tensor): Word-level EEG Feature (batch_size, seq_len, d_model)
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


class BrainTranslatorOG(nn.Module):
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
    """Sanity check."""
    set_seed(888)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model configuration
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    pretrained_seq2seq = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    d_model = 840
    nhead = 8
    dim_feedforward = 2048
    dim_decoder = 1024
    num_layers = 6

    # Create an instance of the BrainTranslator model
    model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    
    model = BrainTranslator(
        pretrained_seq2seq,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dim_decoder=dim_decoder,
        num_layers=num_layers
    )

    # Print the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # random sample from dataset for testing (look at sanity check dataset.py, send to device)
    # assert shape of input to model

    # # Pass the input data through the model
    # out = model(src, input_masks_batch, input_masks_invert, labels)

    # # Print shape of the output and gradient
    # print("Output shape:", out.logits.shape)
    # print("Gradient:", out.logits.grad)
    # decode into string and print

    # # Compare to output and gradients for OG (original) model
    # og_model = YourOriginalModelHere()
    # og_out = og_model(src, input_masks_batch, input_masks_invert, labels)
    # print("Original model output shape:", og_out.shape)
    # print("Original model gradient:", og_out.grad)

    # # Print number of trainable params for both
    # og_num_params = sum(p.numel() for p in og_model.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters in original model: {og_num_params}")


if __name__ == "__main__":
    main()
