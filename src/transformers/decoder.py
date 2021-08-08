import torch
import torch.nn as nn

from transformers.attention import SelfAttention
from transformers.blocks import AddNorm
from transformers.transformer import TransformerBlock


class DecoderSubSection(nn.Module):

    def __init__(
            self,
            embedding_size: int,
            num_heads: int,
            forward_expansion: int,
            dropout_prob: float,
            device: str
    ):
        super(DecoderSubSection, self).__init__()

        self.device = device

        # create the first masked attention section
        self.attention = SelfAttention(embedding_size, num_heads)
        self.add_norm = AddNorm(embedding_size)

        # create the transformer block to ingest the output from the norm layer
        self.transformer = TransformerBlock(embedding_size, num_heads, dropout_prob, forward_expansion)

        self.dropout = nn.Dropout(dropout_prob)

        # finally, move everything to the appropriate device
        self.to(self.device)

    def forward(self, x: torch.Tensor, value_input: torch.Tensor, key_input: torch.Tensor, mask: torch.Tensor):

        # pass through masked self attention section
        attention = self.attention(x, x, x, mask)
        query = self.dropout(self.add_norm(attention, x))

        # pass through the transformer section - using appropriate vectors
        out = self.transformer(value_input, key_input, query)

        return out


