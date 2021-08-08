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
            dropout_prob,
            max_length,
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

