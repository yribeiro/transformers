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

    def forward(self):
        raise NotImplementedError()
