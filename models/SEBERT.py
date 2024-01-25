import torch
import torch.nn as nn
from models.Embedding import Embedding
from pprint import pprint

torch.autograd.set_detect_anomaly(True)


class SENet(nn.Module):
    def __init__(self, model_dim, feature_num, history_len, fusion_method) -> None:
        super(SENet, self).__init__()
        self.model_dim = model_dim
        self.feature_num = feature_num
        self.history_len = history_len

        self.fusion_method = fusion_method
        if self.fusion_method == "mean":
            self.avgpool = nn.AvgPool2d((self.history_len, self.model_dim))
        elif self.fusion_method == "gate":
            self.fusion = nn.Linear(self.model_dim, 1)
            self.avgpool = nn.AvgPool2d((self.history_len, self.model_dim))
        else:
            raise ValueError("fusion_method must be in ('mean', 'gate')")
    
        self.Linear = nn.Sequential(
            nn.Linear(self.feature_num, self.feature_num // 2),
            nn.ReLU(),
            nn.Linear(self.feature_num // 2, self.feature_num),
            nn.ReLU(),
            )
    
    def forward(self, x):
        """
        Params:
            x: [batch_size, feature_num, history_len, model_dim]
        Returns:
            x: [batch_size, feature_num, history_len, model_dim]
        """
        if self.fusion_method == "mean":
            x1 = self.avgpool(x)
        elif self.fusion_method == "gate":
            weight = self.fusion(x)
            x1 = x * weight
            x1 = self.avgpool(x1)
        
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.Linear(x1)
        x2 = x1.unsqueeze(-1).unsqueeze(-1) * x
        return x2


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, feature_num,history_len, fusion_method) -> None:
        super(TransformerBlock, self).__init__()
        self.model_dim = model_dim
        self.feature_num = feature_num
        self.history_len = history_len
        self.fusion_method = fusion_method

        # * MaskedSelfAttention
        self.SENet = SENet(
            model_dim=self.model_dim,
            feature_num=self.feature_num,
            history_len=self.history_len,
            fusion_method=self.fusion_method,
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
        x_1 = self.SENet(x)
        x_1 = nn.functional.dropout(x_1, p=0.7)
        x_1 = x + x_1
        x_2 = nn.functional.layer_norm(x_1, x_1.size()[1:])

        x_2 = self.FeedForward(x_2)
        x_2 = nn.functional.dropout(x_2, p=0.7)
        x_2 = x_1 + x_2
        x_3 = nn.functional.layer_norm(x_2, x_2.size()[1:])

        return x_3


class SEBERT(nn.Module):
    def __init__(self, model_dim, ebd_params, model_params) -> None:
        super(SEBERT, self).__init__()
        self.model_dim = model_dim
        self.ebd_params = ebd_params
        self.model_params = model_params

        self.num_layer = self.model_params['num_layer']
        self.ebd_method = self.model_params['ebd_method']
        self.history_len = self.model_params['history_len']
        self.feature_num = len(self.ebd_params)
        self.fusion_method = self.model_params['fusion_method']
        self.loss_method = self.model_params['loss_method']

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
                feature_num=self.feature_num,
                history_len=self.history_len,
                fusion_method=self.fusion_method,
                )
                for _ in range(self.num_layer)]
            )

        self.BatchNorm1d = nn.BatchNorm1d(self.model_dim)

        # * Prediction
        if self.fusion_method == "gate":
            self.fusion = nn.Linear(self.model_dim, 1)
        else: pass

        in_dim = self.model_dim * (self.feature_num) * 2
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
        if self.ebd_method == "fine-grained":
            history_vec, candidate_vec = self.Embedding(
                method="stack",
                feature_dict=feature_dict,
                )
        else:
            raise ValueError("ebd_method must be in ('fine-grained')")

        history_vec = nn.functional.layer_norm(history_vec, normalized_shape=history_vec.size()[1:])
        candidate_vec = nn.functional.layer_norm(candidate_vec, normalized_shape=candidate_vec.size()[1:])

        # * Construct Block Input: x
        x = torch.cat([history_vec, candidate_vec.unsqueeze(2)], dim=2)
        # [batch_size, feature_num, history_len+1, model_dim]
        x = x + self.PositionalEmbedding
        x = nn.functional.layer_norm(x, x.size()[1:])

        # * TransformerBlock
        for i in range(self.num_layer):
            x = self.TransformerBlockList[i](x)

        # * Prediction
        if self.fusion_method == "mean":
            x = torch.mean(x, dim=2, keepdim=False)
        elif self.fusion_method == "gate":
            weight = self.fusion(x)
            x = torch.sum(x * weight, dim=2, keepdim=False)
        else:
            raise ValueError("fushion_method must be in ('gate','mean')")
        
        # x = self.BatchNorm1d(x)
        x = torch.flatten(x, start_dim=1) # [batch_size, feature_num*model_dim]
        candidate_vec = torch.flatten(candidate_vec, start_dim=1)
        x = torch.concat([x, candidate_vec], dim=-1) # [batch_size, feature_num*model_dim*2]
        logits = self.fc(x)

        # * CrossEntropyLoss
        loss = self.loss(logits, candidate_label.to(torch.long))

        return logits, loss
