import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=16, hid_layers_dims=[32, 16, 8], dropout_rate=0.2, init=False):
        super(MLP, self).__init__()
        
        self.user_embedding = nn.Embedding(num_embeddings=n_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=n_items, embedding_dim=embedding_dim)

        layers = []
        input_dim = embedding_dim * 2

        for dim in hid_layers_dims:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = dim

        layers.append(nn.Linear(hid_layers_dims[-1], 1))

        self.layers = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout_rate)

        if init:
            self.user_embedding.weight.data.uniform_(-0.1, 0.1)
            self.item_embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, user_input, item_input):
        user_embedded = self.dropout(self.user_embedding(user_input))
        item_embedded = self.dropout(self.item_embedding(item_input))

        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        output = self.layers(vector)

        return output.squeeze()

