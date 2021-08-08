import torch
import torch.nn as nn

from transformers.transformer import TransformerBlock


class Encoder(nn.Module):
    """
    The encoder block as defined in the paper. The block is responsible for taking an
    input embeddings sequence and applying the appropriate attention weightings on the
    input words based on the output sentence.
    """

    def __init__(
            self,
            vocab_size: int,
            embedding_size: int,
            num_layers: int,
            num_heads: int,
            forward_expansion,
            max_length,
            dropout_prob=0.5,
            device: str = "cpu"
    ):
        super(Encoder, self).__init__()

        self.embed_size = embedding_size
        self.device = device

        # create the embedding layer - this layer is responsible for learning word embeddings
        # since there are predefined word embedding models already learnt - this layer might
        # use an existing embedding and freeze during training
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)

        # Query: create a position embedding - this is different from the paper
        # the paper uses sines and cosines to create a sense of positioning in a sentence
        # not sure why in this implementation this is being learned
        self.position_embedding = nn.Embedding(max_length, embedding_size)

        # create the Transformer block section of the encoder with multiple layers
        self.transformer_section = nn.ModuleList(
            [TransformerBlock(embedding_size, num_heads, dropout_prob, forward_expansion) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout_prob)

        # send the entire module onto the device - which should take all modules and move to appropriate
        # memory
        self.to(self.device)

    def forward(self, x, mask):
        batch_size, seq_length = x.shape
        # this creates a batch_size of vectors that contain numbers from 0 -> seq_length
        # in combination with the learnable position embeddings in the constructor create
        # position awareness in the model - still do not understand how this is happening!
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # pass through the transformer section after applying position encodings
        for transformer in self.transformer_section:
            # in the encoder the input (query, key and value vectors) are all the same
            out = transformer(out, out, out, mask)
            
        return out
