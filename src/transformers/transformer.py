from typing import Optional

import torch
import torch.nn as nn

from transformers.attention import SelfAttention
from transformers.blocks import AddNorm


class TransformerBlock(nn.Module):
    """
    The Transformer block as defined in the paper.

    The block pulls together the multi-headed self attention block, along with skip connections and
    a feed forward layer for extra processing.
    """

    def __init__(self, embedding_size: int, num_heads: int, dropout_prob: float, forward_expansion: int):
        """
        Constructor

        :param embedding_size: Number of dimensions in the embedding vector used to encode words.
        :param num_heads: Number of heads to include in the self-attention block.
        :param dropout_prob: Probability with which to dropout weights during learning.
        :param forward_expansion:
            Term used from paper to indicate the hidden layer size in the FC layer after the multi-headed attention
            block is used.
        """
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embedding_size, num_heads)

        # normalisation layer
        self.add_norm1 = AddNorm(embedding_size)
        self.add_norm2 = AddNorm(embedding_size)

        # this component allows flexibility to learn more complicated relationships
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size)
        )

        self.dropout = nn.Dropout(dropout_prob)

    def forward(
            self, value_input: torch.Tensor, key_input: torch.Tensor, query_input: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Method to run inference using the Transformer block.

        :param value_input: Vector for the value input.
        :param key_input: Vector for the key input.
        :param query_input: Vector for the query input.
        :param mask: Optional mask to be applied to the embedding vectors prior to the softmax in attention.
        :return: Output vector containing the attention weighted embeddings for the sequence.
        """
        embedding_with_attention = self.attention(value_input, key_input, query_input, mask)

        x = self.dropout(self.add_norm1(embedding_with_attention, query_input))
        out = self.feed_forward(x)
        out = self.dropout(self.add_norm2(out, x))

        return out
