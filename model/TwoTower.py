import torch
import torch.nn as nn
from model.BaseModel import BaseModel


class TwoTower(BaseModel):
    def __init__(self,conf):
        super(TwoTower,self).__init__()
        self.user_embedding_num = conf["user_embedding_num"]
        self.user_embedding_dim = conf["user_embedding_dim"]

        self.item_embedding_num = conf["item_embedding_num"]
        self.item_embedding_dim = conf["item_embedding_dim"]
        self.user_embedding_dim2 = 200
        self.item_embedding_dim2 = 200

        self.user_dense = [len(conf["user_features"])*self.user_embedding_dim ,*conf["user_dense"]]
        self.item_dense = [len(conf
                               ["item_features"])*self.item_embedding_dim ,*conf["item_dense"]]
        self.activation = conf["activation"]
        user_dense_layers = []
        item_dense_layers = []
        self.flatten = nn.Flatten()
        self.user_embedding = nn.Embedding(self.user_embedding_num + 1,self.user_embedding_dim)
        for i  in range(len(self.user_dense) - 1):
            dense = nn.Linear(self.user_dense[i],self.user_dense[i+1])
            if self.activation == "relu":
                act = nn.ReLU()

            user_dense_layers.append(dense)
            user_dense_layers.append(act)

        self.user_tower = nn.Sequential(*user_dense_layers)

        self.item_embedding = nn.Embedding(self.item_embedding_num + 1,self.item_embedding_dim)
        for i  in range(len(self.item_dense) - 1):
            dense = nn.Linear(self.item_dense[i],self.item_dense[i+1])
            if self.activation == "relu":
                act = nn.ReLU()

            item_dense_layers.append(dense)
            item_dense_layers.append(act)

        self.item_tower = nn.Sequential(*item_dense_layers)
        
    def forward(self,X):
        user_embed = self.user_embedding(X[0])
        user_embed = self.flatten(user_embed)
        item_embed = self.item_embedding(X[1])
        item_embed = self.flatten(item_embed)

        user  = self.user_tower(user_embed)
        item  = self.item_tower(item_embed)
    
        score = torch.dot(user.reshape((-1,)), item.reshape((-1,)))
        return score
    