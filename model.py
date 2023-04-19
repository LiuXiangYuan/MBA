import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, user_num, item_num, K0):
        super(MF, self).__init__()
        self.user_embedding = nn.Embedding(user_num, K0)
        self.item_embedding = nn.Embedding(item_num, K0)
        self._init_weight_()

    def forward(self, user, item):
        # user: batch_size
        # item: batch_size
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        p = torch.bmm(user_embedding.unsqueeze(1), item_embedding.unsqueeze(1).transpose(1, 2)).reshape(-1)
        return p

    def _init_weight_(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.user_embedding(users)
        items_emb = self.item_embedding.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return torch.sigmoid(scores)
