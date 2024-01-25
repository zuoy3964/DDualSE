import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn import metrics as metrics
from sklearn.metrics import ndcg_score, roc_auc_score
from tqdm import tqdm
from itertools import chain
from collections import Counter


# Initialization
pandarallel.initialize(progress_bar=False)

# cal metrics for each user, and then average them
def flatten(nested_list: list) -> list:
    if str(nested_list[0]).isdigit():
        flattened_list = nested_list
    else:
        flattened_list = list(chain.from_iterable(nested_list))
    return flattened_list


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def diversity_score(y_score, candidate_category, k=5):
    # pick top 5 items to calculate entropy
    order = np.argsort(y_score)[::-1][:k]
    candidate_category = np.take(candidate_category, order)
    counts = Counter(candidate_category)
    total_count = len(candidate_category)
    probabilities = np.array(list(counts.values())) / total_count
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

def hit_rate_score(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1][:k]
    y_true = np.take(y_true, order)
    return np.sum(y_true) / k


def calculate_metrics(truth, pred_rank):
    """
    Desc: Cal metrics for each user
    Args:
        truth: ground-truth label，[1,0,1,0,1]
        pred_rank: predicted ranking，[2,1,3,5,4]
    Returns:
        auc: auc score
        ndcg5: ndcg@5 score
        ndcg10: ndcg@10 score
        mrr: mrr score
    """
    y_true = np.array(truth, dtype='float32')
    y_score = np.array(1 / pred_rank, dtype='float32')

    # print("len(np.unique(y_true))=",len(np.unique(y_true)))
    if len(np.unique(y_true)) < 2:  # if there is only one class, then AUC can't be calculated
        return -1, -1, -1, -1, -1, -1
    else:
        auc = roc_auc_score(y_true, y_score)
        mrr = mrr_score(y_true, y_score)
        ndcg5 = ndcg_score(y_true, y_score, 5)
        ndcg10 = ndcg_score(y_true, y_score, 10)
        return auc, ndcg5, ndcg10, mrr


def cal_avg_metrics(user_id_list, candidate_label_list, softmax_list, prediction_list, candidate_list, **kwargs):

    df = pd.DataFrame({
        'user_id': user_id_list,
        'candidate_label': candidate_label_list,
        'click_prob': softmax_list,
        'prediction': prediction_list,
        'candidate_id': candidate_list,
        })

    df["ranking"] = df.groupby("user_id")["click_prob"].rank("dense", ascending=False)
    df['ranking'] = df['ranking'].astype('int64')

    auc, ndcg5, ndcg10, mrr = zip(*df.groupby('user_id')[['candidate_label', 'ranking']].parallel_apply(
        lambda x: calculate_metrics(x['candidate_label'], x['ranking'])))

    auc = np.array(auc)
    ndcg5 = np.array(ndcg5)
    ndcg10 = np.array(ndcg10)
    mrr = np.array(mrr)

    auc = np.mean(auc[auc != -1])
    ndcg5 = np.mean(ndcg5[ndcg5 != -1])
    ndcg10 = np.mean(ndcg10[ndcg10 != -1])
    mrr = np.mean(mrr[mrr != -1])

    return {"AUC": auc, "nDCG@5": ndcg5, "nDCG@10": ndcg10, "MRR": mrr}
