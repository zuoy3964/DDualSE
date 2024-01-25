import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, ebd_params: list, model_dim: int):
        super(Embedding, self).__init__()

        self.input_set = set(ebd_params)
        self.model_dim = model_dim
        self.feature_num = len(ebd_params)

        self.EmbeddingDict = nn.ModuleDict()
        for data_category in self.input_set:
            if 'text' in data_category:
                dim = int(data_category.split('_')[-1])
                self.EmbeddingDict[data_category] = nn.Linear(
                    in_features=dim,
                    out_features=self.model_dim,
                    bias=False
                    )
            elif "categorial" in data_category:
                dim = int(data_category.split('_')[-1])
                self.EmbeddingDict[data_category] = nn.Embedding(
                    num_embeddings=dim,
                    embedding_dim=self.model_dim,
                    padding_idx=0,
                    )
            elif "sequential" in data_category:
                dim = int(data_category.split('_')[-1])
                self.EmbeddingDict[data_category] = nn.Embedding(
                    num_embeddings=dim,
                    embedding_dim=self.model_dim,
                    padding_idx=dim - 1,  # the last item is the padding item
                    )
            else:
                raise ValueError(
                    "the key of ebd_params should be ('text_x', 'categorial_x' or 'sequential_x'), while 'x' should "
                    "be the original dim! but got ({})".format(
                        self.input_set
                        )
                    )

        self.model_dim_linear = nn.Linear(
            in_features=self.model_dim,
            out_features=self.model_dim,
            )

        self.concat_linear = nn.Linear(
            in_features=self.model_dim * self.feature_num,
            out_features=self.model_dim * self.feature_num,
            )

    def forward(self, feature_dict, method):
        """
        Params:
            feature_dict: a dict of features, key is the feature name, value is the feature value
            method: 
                concat: concat all features at dim -1, output:[batch_size, history_len, model_dim*feature_num]
                stack: stack all features at dim 1, output:[batch_size, feature_num, history_len, model_dim]
                sum: sum all features at feature dim, output:[batch_size, history_len, model_dim]
        Return:
            history_vec: history feature matrix
            candidate_vec: candidate feature matrix
        """
        feature_dict_new = {}
        for data_category, features in feature_dict.items():
            for feature_name, feature_value in features.items():
                if "text" in data_category:
                    feature_dict_new[feature_name] = self.EmbeddingDict[data_category](feature_value.to(torch.float32))
                elif "categorial" in data_category:
                    feature_dict_new[feature_name] = self.EmbeddingDict[data_category](feature_value.to(torch.long))
                elif "sequential" in data_category:
                    feature_dict_new[feature_name] = self.EmbeddingDict[data_category](feature_value.to(torch.long))
                else:
                    raise ValueError(
                        "the key of ebd_params should be ('text_x', 'categorial_x' or 'sequential_x'), while 'x' "
                        "should be the original dim! but got ({})".format(
                            self.input_set
                            )
                        )

        # concat or stack features

        if method == "concat":  # [batch_size, history_len, model_dim*feature_num]
            history_vec = torch.cat(
                [v for k, v in feature_dict_new.items() if 'history' in k],
                dim=-1
                )
            candidate_vec = torch.cat(
                [v for k, v in feature_dict_new.items() if 'candidate' in k],
                dim=-1
                )
            history_vec = self.concat_linear(history_vec)
            candidate_vec = self.concat_linear(candidate_vec)
        elif method == "stack":  # [batch_size, feature_num, history_len, model_dim]
            history_vec = torch.stack(
                [v for k, v in feature_dict_new.items() if 'history' in k],
                dim=1
                )
            candidate_vec = torch.stack(
                [v for k, v in feature_dict_new.items() if 'candidate' in k],
                dim=1
                )
            history_vec = self.model_dim_linear(history_vec)
            candidate_vec = self.model_dim_linear(candidate_vec)
        elif method == 'sum':  # [batch_size, history_len, model_dim]
            history_vec = torch.sum(
                torch.stack(
                    [v for k, v in feature_dict_new.items() if 'history' in k],
                    dim=1
                    ),
                dim=1
                )
            candidate_vec = torch.sum(
                torch.stack(
                    [v for k, v in feature_dict_new.items() if 'candidate' in k],
                    dim=1
                    ),
                dim=1
                )
            history_vec = self.model_dim_linear(history_vec)
            candidate_vec = self.model_dim_linear(candidate_vec)
        elif method == 'none':  # [batch_size, history_len, model_dim]
            # if it is the sequential feature, we should return the original feature
            history_vec = feature_dict_new['history_seq']
            candidate_vec = feature_dict_new['candidate_seq']
            history_vec = self.model_dim_linear(history_vec)
            candidate_vec = self.model_dim_linear(candidate_vec)
        else:
            raise ValueError("method should be in ('concat','stack','sum','none')!")

        return history_vec, candidate_vec
