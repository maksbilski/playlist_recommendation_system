import torch
import torch.nn as nn


class WMF(nn.Module):
    def __init__(self, 
                 n_users: int,
                 n_items: int,
                 embedding_dim: int = 8,
                 dropout_rate:float = 0.2,
                 init: bool = False,
                 bias: bool = False,
                 sigmoid: bool = False
                 ) -> None:
        super(WMF, self).__init__()

        self.bias = bias
        self.sigmoid = sigmoid
        self.user_embedding = nn.Embedding(num_embeddings=n_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=n_items, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

        if bias:
            self.user_bias = nn.Parameter(torch.zeros(n_users))
            self.item_bias = nn.Parameter(torch.zeros(n_items))
            self.offset = nn.Parameter(torch.zeros(1))

        if init:
            self.user_embedding.weight.data.uniform_(-0.1, 0.1)
            self.item_embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        user_embedded = self.dropout(self.user_embedding(user_input))
        item_embedded = self.dropout(self.item_embedding(item_input))

        element_product = (user_embedded * item_embedded).sum(dim=1)

        if self.bias:
            element_product += self.user_bias[user_input] + self.item_bias[item_input] + self.offset

        if self.sigmoid:
            element_product = torch.sigmoid(element_product)

        return element_product.squeeze()
