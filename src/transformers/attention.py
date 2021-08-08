from typing import Optional

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    The self attention block:
        - takes in word embeddings
        - generates attention weights (query map x key map)
        - weights the values i.e. softmax(QK^T) * V

    The block can have multiple heads running in parallel. This implicitly splits the embedding
    vector and process chunks across each of the heads. Weighting each element in the vector
    based on the amount of attention to pay.

    The heads are independent and are processed in parallel. The outputs of the head are the combined to
    reform the embedding vector but with the attention information encoding for each word.
    """

    def __init__(self, embedding_size: int, num_heads: int):
        """
        Constructor

        :param embedding_size: Number of dimensions in the embedding vector used to encode words.
        :param num_heads: Number of heads to include in the self-attention block.
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.num_heads = num_heads

        # the attention block splits the embedding vector evenly amongst all the heads
        # as a result, the embedding needs to be evenly divisible by the num_heads
        assert embedding_size % num_heads == 0, "The embedding size needs to be evenly divisible by the number of heads"
        self.head_size = embedding_size // num_heads

        # setup the layers for values, keys and queries
        # these layers are simple matrix maps i.e. inp * W = out
        # where inp: [1 x self.head_size], W: [self.head_size, self.head_size], out: [1 x self.head_size]

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)

        # this is the layer after concatenation i.e. num_heads * head_size
        self.fc_out = nn.Linear(embedding_size, embedding_size)

    def forward(
            self, value_input: torch.Tensor, key_input: torch.Tensor, query_input: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Method to run inference using the model.

        :param value_input: Vector for the value input in attention.
        :param key_input: Vector for the key input in attention.
        :param query_input: Vector for the query input in attention.
        :param mask:
            Optional mask of ones and zeroes. The algorithm will set all elements in the energy tensor, whose indices
            correspond to 1s in the mask, to a very small number. This results a near 0 value, after the
            softmax function is applied to the tensor.
        :return:
            The batch_size x sequence length x embedding vector len tensor that contains attention embedded word
            encodings.
        """
        err_msg = "the query needs to have a dimension of 3 i.e. batch_size x seq_length x embedding_size"
        assert len(query_input.shape) == 3, err_msg
        batch_size = query_input.shape[0]

        # this retrieves the vector size for the input vectors
        val_seq_len, key_seq_len, query_seq_len = value_input.shape[1], key_input.shape[1], query_input.shape[1]

        # reshape the embedding to run across the number of heads
        segmented_val = value_input.view(batch_size, val_seq_len, self.num_heads, self.head_size)
        segmented_keys = key_input.view(batch_size, key_seq_len, self.num_heads, self.head_size)
        segmented_queries = query_input.view(batch_size, query_seq_len, self.num_heads, self.head_size)

        # use torch.einsum to run the query x key step - we could optionally do a batch matrix multiply
        # the dimensions in the tensors are [batch_size, seq_length, num_heads, head_size]
        # the nhqk outlines the dimensions of the energy tensor
        # i.e. energy output dims is [batch_size, num_heads, query_seq_len, key_seq_len]
        energy = torch.einsum("nqhd,nkhd->nhqk", [segmented_queries, segmented_keys])

        # optionally apply a mask
        if mask is not None:
            # replace all the elements in energy with negative infinity where the mask is 0
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # run softmax on the key dimension in energy i.e. [batch_size, num_heads, query_seq_len, key_seq_len]
        # this essentially says how much attention do I want to pay to each input word in the input sequence
        # for every word in the output sequence - BUT in each dimension of the embedding vector of each word
        attention = torch.softmax(energy / self.embedding_size ** 0.5, dim=3)

        # attention dims: [batch_size, num_heads, query_seq_len, key_seq_len]
        # values dims: [batch_size, seq_length, num_heads, head_size]

        # the key and value seq_length will always be the same
        # we eventually want to get back to the embedding vector i.e. [batch_size, seq_length, num_heads, head_size]

        # Query: still unsure why we are using the query len - might need to do the explicit matrix calc
        # to figure out the order of operations
        out = torch.einsum("nhql,nlhd->nqhd", [attention, segmented_val])

        # run the concatenation step to recover the embedding vectors for words
        # we could use -1 as the last dimension here, but this implementation will throw an error if we
        # cannot recover the embedding size, which is what we want
        out = out.view(batch_size, query_seq_len, self.embedding_size)
        return out
