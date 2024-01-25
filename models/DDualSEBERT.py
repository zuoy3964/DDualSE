import numpy as np
import torch
import torch.nn as nn
from models.Embedding import Embedding
from pprint import pprint

np.set_printoptions(threshold=1000000)
torch.set_printoptions(threshold=torch.inf)


class DDualSE(nn.Module):
    def __init__(self, model_dim, num_head, fusion_method, feature_num, history_len, mask):
        super(DDualSE, self).__init__()

        self.num_head = num_head
        self.model_dim = model_dim
        self.feature_num = feature_num
        self.history_len = history_len
        self.mask = mask
        self.fusion_method = fusion_method
        if self.fusion_method == "gate":
            self.fusion = nn.Linear(self.model_dim, 1)
        else:
            pass

        while self.model_dim % self.num_head != 0:
            self.num_head -= 1

        in_dim = self.feature_num+self.history_len
        self.q = nn.Linear(in_dim,in_dim)
        self.k = nn.Linear(in_dim,in_dim)
        self.v = nn.Linear(self.model_dim, self.model_dim)
        
        # self.W_A = nn.Linear(self.model_dim//self.num_head, 1, bias=False)
        self.W_O = nn.Linear(self.model_dim, self.model_dim)

        
    def forward(self, x):
        """
        Params:
            x: [batch_size(B), feature_num(F), history_len(H), model_dim(D)]
        Returns:
            v: [batch_size, feature_num, history_len, model_dim]
        """

        if self.fusion_method == "sum":
            x_history = torch.sum(x, dim=1, keepdim=False)
            x_feature = torch.sum(x, dim=2, keepdim=False)
        elif self.fusion_method == "mean":
            x_history = torch.mean(x, dim=1, keepdim=False)  # [batch_size, history_len, model_dim]
            x_feature = torch.mean(x, dim=2, keepdim=False)  # [batch_size, feature_num, model_dim]
        elif self.fusion_method == "gate":
            weight = torch.sigmoid(self.fusion(x))  # [batch_size, feature_num, history_len, 1]
            x = x * weight
            x_history = torch.sum(x, dim=1, keepdim=False)
            x_feature = torch.sum(x, dim=2, keepdim=False)
        else:
            raise ValueError("fusion_method must be in ('sum','mean','gate')")

        B, H, D = x_history.shape
        B, F, D = x_feature.shape

        q = x_feature.reshape(B, self.num_head, F, -1)
        k = x_history.reshape(B, self.num_head, H, -1)
        o = torch.concat([q, k], dim=2) # [B,head,H+F,D]

        q = torch.matmul(q, o.transpose(-1, -2))/ (D ** 0.5)# [B,head,F,H+F]
        k = torch.matmul(k, o.transpose(-1, -2))/ (D ** 0.5)# [B,head,H,H+F]

        # print("q.shape=",q.shape)
        # print("k.shape=",k.shape)
        q = self.q(q)
        k = self.k(k)
        att = torch.matmul(q, k.transpose(-1, -2))/ (D ** 0.5) # [B,head,F,H]
        att = nn.functional.softmax(att, dim=-2) # take softmax over history_len

        # v = self.v(x).reshape(B, self.num_head, F, H, -1)
        v = x.reshape(B, self.num_head, F, H, -1)
        v = att.unsqueeze(-1) * v

        # fusion multiheads
        v = v.reshape(B, F, H, -1)
        v = self.W_O(v)
        
        return v, att


class MultiHeadSEBlock(nn.Module):
    def __init__(self, model_dim, num_head, mask, fusion_method, feature_num, history_len) -> None:
        super(MultiHeadSEBlock, self).__init__()

        self.model_dim = model_dim
        self.fusion_method = fusion_method
        self.mask = mask
        self.feature_num = feature_num
        self.history_len = history_len
        self.num_head = num_head
        self.DDualSE = DDualSE(
            model_dim=self.model_dim,
            num_head=self.num_head,
            mask=self.mask,
            fusion_method=self.fusion_method,
            feature_num=self.feature_num,
            history_len=self.history_len,
            )

        # * FeedForward
        self.FeedForward = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.ReLU(),
            nn.Linear(self.model_dim, self.model_dim)
            )

    def forward(self, x):
        """
        Params:
            x: [batch_size, feature_num, history_len, model_dim]
        returns:
            x: [batch_size, feature_num, history_len, model_dim]
        """
        x_1, att = self.DDualSE(x)
        x_1 = nn.functional.dropout(x_1, p=0.5)
        x_1 = x + x_1
        x_2 = nn.functional.layer_norm(x_1, x_1.size()[1:])

        x_2 = self.FeedForward(x_2)
        x_2 = nn.functional.dropout(x_2, p=0.5)
        x_2 = x_1 + x_2
        x_3 = nn.functional.layer_norm(x_2, x_2.size()[1:])

        return x_3, att



class DDualSEBERT(nn.Module):
    def __init__(self, model_dim, ebd_params, model_params) -> None:
        super(DDualSEBERT, self).__init__()
        self.model_dim = model_dim
        self.ebd_params = ebd_params
        # find the number of category features

        self.model_params = model_params

        self.num_head = int(self.model_params['num_head'])
        self.num_layer = self.model_params['num_layer']
        self.ebd_method = self.model_params['ebd_method']
        self.feature_num = len(self.ebd_params)
        self.history_len = self.model_params['history_len']
        self.fusion_method = self.model_params['fusion_method']
        self.loss_method = self.model_params['loss_method']
        
        print("\nmodel_params:")
        pprint(model_params)

        self.Embedding = Embedding(
            ebd_params=self.ebd_params,
            model_dim=self.model_dim,
            )

        self.PositionalEmbedding = nn.Parameter(
            torch.randn(self.history_len + 1, self.model_dim)
            )

        self.TransformerBlockList = nn.ModuleList(
            [
                MultiHeadSEBlock(
                    model_dim=self.model_dim,
                    num_head=self.num_head,
                    mask=False,
                    fusion_method=self.fusion_method,
                    feature_num=self.feature_num,
                    history_len=self.history_len+1,
                    )
                for _ in range(self.num_layer)
                ]
            )

        # * Prediction
        if self.fusion_method == "gate":
            self.fusion1 = nn.Linear(self.model_dim, 1)
            self.fusion2 = nn.Linear(self.model_dim, 1)
        else:
            pass
        # in_dim = self.model_dim * (self.history_len + 1)
        in_dim = self.model_dim * self.feature_num * 2
        # in_dim = self.model_dim * 2
        # in_dim = self.model_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_dim // 4, 2),
            )

        self.lambda_ = nn.Parameter(torch.tensor(0.1))

    def loss(self, logits, candidate):
        if self.loss_method.lower() == "crossentropy":
            return nn.functional.cross_entropy(logits, candidate.to(torch.long))
        elif self.loss_method == "focal":
            loss = nn.functional.cross_entropy(logits, candidate.to(torch.long))
            p = torch.exp(-loss)
            focal_loss = self.alpha * (1 - p) ** self.gamma * loss
            return focal_loss.mean()
        else:
            raise ValueError("loss_method must be in ('crossentropy','focal')")
    
    def aux_loss(self,att):
        B,N,H,D = att.shape
        att = torch.flatten(att, start_dim=2)
        att = torch.bmm(att, att.transpose(1, 2))/((H*D)**0.5)
        att = torch.triu(att, diagonal=1)
        loss = torch.flatten(att, start_dim=1)
        return torch.mean(loss)
        

    def forward(self, candidate_label, feature_dict):

        # * Embedding
        if self.ebd_method == "fine-grained":
            history_vec, candidate_vec = self.Embedding(
                method="stack",
                feature_dict=feature_dict,
                )
        else:
            raise ValueError("ebd_method must be in ('fine-grained')")

        x = torch.cat([history_vec, candidate_vec.unsqueeze(2)], dim=2)
        # [batch_size, feature_num, history_len+1, model_dim]
        x = x + self.PositionalEmbedding
        x = nn.functional.layer_norm(x, x.size()[1:])
        # [batch_size, feature_num, history_len, model_dim]

        # * TransformerBlock
        for i in range(self.num_layer):
            x, att = self.TransformerBlockList[i](x=x)

        # * Prediction
        # gathering the history -> [batch_size, feature_num, model_dim]
        if self.fusion_method == "sum":
            x = torch.sum(x, dim=2, keepdim=False)
        elif self.fusion_method == "mean":
            x = torch.mean(x, dim=2, keepdim=False)
        elif self.fusion_method == "gate":
            weight = torch.sigmoid(self.fusion2(x))
            x = x * weight
            x = torch.sum(x, dim=2, keepdim=False)
        else:
            raise ValueError("fusion_method must be in ('sum','mean','gate')")

        x = torch.flatten(x, start_dim=1)
        candidate_vec = torch.flatten(candidate_vec, start_dim=1)
        x = torch.concat([x, candidate_vec], dim=-1) # [batch_size, feature_num*model_dim*2]
        logits = self.fc(x)

        # * Loss
        loss = self.loss(logits, candidate_label.to(torch.long))
        return logits, loss
