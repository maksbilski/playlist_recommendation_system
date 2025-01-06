import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=8, dropout_rate=0.2, init=False, bias=False, sigmoid=False):
        super(GMF, self).__init__()

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
            self.user_embedding.weight.data.uniform_(0., 0.05)
            self.item_embedding.weight.data.uniform_(0., 0.05)

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        element_product = (user_embedded * item_embedded).sum(dim=1)

        if self.bias:
            element_product += self.user_bias[user_input] + self.item_bias[item_input] + self.offset

        if self.sigmoid:
            element_product = torch.sigmoid(element_product)

        return element_product.squeeze()
