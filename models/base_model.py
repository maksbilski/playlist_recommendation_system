import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=8):
        super(GMF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_embeddings=n_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=n_items, embedding_dim=embedding_dim)
        
        self.output_layer = nn.Linear(in_features=embedding_dim, out_features=1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        
        element_product = torch.mul(user_embedded, item_embedded)
        
        output = self.output_layer(element_product)
        
        prediction = self.sigmoid(output)
        
        return prediction.squeeze()
