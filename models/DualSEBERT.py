import torch
import torch.nn as nn
from models.Embedding import Embedding
from pprint import pprint

torch.autograd.set_detect_anomaly(True)

class MultiHeadAttention3D(nn.Module):
    def __init__(self, model_dim, num_head, fusion_method, mask):
        super(MultiHeadAttention3D, self).__init__()

        self.num_head = num_head
        self.model_dim = model_dim
        self.mask = mask
        self.fusion_method = fusion_method
        if self.fusion_method == "gate":
            self.fusion = nn.Linear(self.model_dim, 1)

        while self.model_dim % self.num_head != 0:
            self.num_head -= 1

        self.q = nn.Linear(self.model_dim, self.model_dim)
        self.k = nn.Linear(self.model_dim, self.model_dim)
        self.v = nn.Linear(self.model_dim, self.model_dim)

        self.W_O = nn.Linear(self.model_dim, self.model_dim)

    def forward(self, x):
        """
        Params:
            x: [batch_size(B), feature_num(F), history_len(H), model_dim(D)]
        Returns:
            v: [batch_size, feature_num, history_len, model_dim]
        """
        B, F, H, D = x.shape
        if self.fusion_method == "gate":
            weight = torch.sigmoid(self.fusion(x))
            x = x * weight
            q = torch.sum(x, dim=2, keepdim=False) # [batch_size, history_len, model_dim]
            k = torch.sum(x, dim=1, keepdim=False) # [batch_size, feature_num, model_dim]
        elif self.fusion_method == "sum":
            q = torch.sum(x, dim=2, keepdim=False)
            k = torch.sum(x, dim=1, keepdim=False)
        elif self.fusion_method == "mean":
            q = torch.mean(x, dim=2, keepdim=False)
            k = torch.mean(x, dim=1, keepdim=False)
        else:
            raise ValueError("fusion_method must be in ('sum','mean','gate')")

        q = self.q(q).reshape(B, self.num_head, F, -1)
        k = self.k(k).reshape(B, self.num_head, H ,-1)
        v = self.v(x).reshape(B, self.num_head, F, H, -1)

        attn = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        if self.mask:
            mask = torch.triu(torch.ones_like(attn), diagonal=1)
            attn = attn.masked_fill(mask == 1, -1e9)
        else:
            pass
        attn = torch.nn.functional.softmax(attn,dim=-2) # [batch_size, num_head, feature_num, history_len]
        
        v = attn.unsqueeze(-1) * v
        v = v.reshape(B,F,H,D)
        v = self.W_O(v)

        return v


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_head, fusion_method, mask) -> None:
        super(TransformerBlock, self).__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        self.fusion_method = fusion_method
        self.mask = mask

        # * MaskedSelfAttention
        self.MultiHeadAttention3D = MultiHeadAttention3D(
            model_dim=self.model_dim,
            num_head=self.num_head,
            fusion_method=fusion_method,
            mask=mask
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
            x: [batch_size, history_len, model_dim]
        returns: 
            x_3: [batch_size, history_len, model_dim]
        """
        x_1 = self.MultiHeadAttention3D(x)
        x_1 = nn.functional.dropout(x_1, p=0.7)
        x_1 = x + x_1
        x_2 = nn.functional.layer_norm(x_1, x_1.size()[1:])

        x_2 = self.FeedForward(x_2)
        x_2 = nn.functional.dropout(x_2, p=0.7)
        x_2 = x_1 + x_2
        x_3 = nn.functional.layer_norm(x_2, x_2.size()[1:])

        return x_3


class DualSEBERT(nn.Module):
    def __init__(self, model_dim, ebd_params, model_params) -> None:
        super(DualSEBERT, self).__init__()
        self.model_dim = model_dim
        self.ebd_params = ebd_params
        self.model_params = model_params

        self.num_layer = self.model_params['num_layer']
        self.num_head = self.model_params['num_head']
        self.ebd_method = self.model_params['ebd_method']
        self.history_len = self.model_params['history_len']
        self.feature_num = len(self.ebd_params)
        self.fusion_method = self.model_params['fusion_method']
        self.mask = self.model_params['mask']

        print("\nmodel_params:")
        pprint(model_params)

        # * Embedding
        self.Embedding = Embedding(
            ebd_params=self.ebd_params,
            model_dim=self.model_dim,
            )

        self.PositionalEmbedding = nn.Parameter(
            torch.randn(self.history_len + 1, self.model_dim)
            )

        # * Transformer
        self.TransformerBlockList = nn.ModuleList(
            [TransformerBlock(
                model_dim=self.model_dim,
                num_head=self.num_head,
                fusion_method=self.fusion_method,
                mask=self.mask
                )
                for _ in range(self.num_layer)]
            )

        self.BatchNorm1d = nn.BatchNorm1d(self.model_dim)

        # * Prediction
        self.fusion = nn.Linear(self.model_dim, 1)
        in_dim = self.model_dim * self.feature_num * 2
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Dropout(0.7),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.Dropout(0.7),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, 2),
            )
        
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

    def forward(self, candidate_label, feature_dict):

        # * Embedding        
        assert self.ebd_method == "fine-grained", "ebd_method must be in ('fine-grained')"
        history_vec, candidate_vec = self.Embedding(
                method="stack",
                feature_dict=feature_dict,
                )

        history_vec = nn.functional.layer_norm(history_vec, normalized_shape=history_vec.size()[1:])
        candidate_vec = nn.functional.layer_norm(candidate_vec, normalized_shape=candidate_vec.size()[1:])

        # * Construct Block Input: x
        x = torch.cat([history_vec, candidate_vec.unsqueeze(2)], dim=2)
        x = x + self.PositionalEmbedding
        x = nn.functional.layer_norm(x, x.size()[1:])

        # * TransformerBlock
        for i in range(self.num_layer):
            x = self.TransformerBlockList[i](x)

        # * Prediction
        # gathering the history -> [batch_size, feature_num, model_dim]
        if self.fusion_method == "sum":
            x = torch.sum(x, dim=2, keepdim=False)
        elif self.fusion_method == "mean":
            x = torch.mean(x, dim=2, keepdim=False)
        elif self.fusion_method == "gate":
            weight = torch.sigmoid(self.fusion(x))
            x = x * weight
            x = torch.sum(x, dim=2, keepdim=False)
        else:
            raise ValueError("fusion_method must be in ('sum','mean','gate')")
        
        x = torch.flatten(x, start_dim=1)
        candidate_vec = torch.flatten(candidate_vec, start_dim=1)
        x = torch.concat([x, candidate_vec], dim=-1) # [batch_size, feature_num*model_dim*2]
        logits = self.fc(x)
        loss = self.loss(logits, candidate_label.to(torch.long))

        return logits, loss
