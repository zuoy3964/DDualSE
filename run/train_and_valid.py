import torch
from metrics import flatten, cal_avg_metrics
from datetime import datetime
from get_feature_dict import get_feature_dict
from pprint import pprint

def train_or_valid(
        dataset_dict: dict,
        model_params: dict,
        model,
        mode: str,
        epoch: int,
        dataloader,
        optimizer,
        device,
        stop_step=None,
        ):
    """
    Desc:
        train or valid
    Args:
        dataset_dict: dataset_dict
        model_params: model_params
        model: model
        mode: train or valid
        epoch: epoch current
        dataloader
        optimizer
        device
        stop_step: when step > stop_step, break
    Returns:
        model
        optimizer
    """
    dataset = dataset_dict['dataset_name']
    model_name = model_params['model_name']

    if mode == 'train':
        model.train()
    elif mode in ('valid', 'test'):
        model.eval()
    else:
        raise AttributeError("mode should be train/valid/test!")

    loss = 0
    user_list = []
    softmax_list = []
    prediction_list = []
    candidate_list = []
    candidate_label_list = []

    # start_time = datetime.now()
    for i, data in enumerate(dataloader):

        # print(f"loading time={(datetime.now()-start_time).seconds}s!")
        if mode == 'train':
            optimizer.zero_grad()
        else:
            pass

        uid = data[0]
        if dataset == "bookcrossing":
            candidate_label = data[13].to(device)
            candidate_list.append(data[7].detach().cpu().tolist())
        elif dataset in ('mindlarge', 'mindsmall'):
            candidate_label = data[10].to(device)
            candidate_list.append(data[5].detach().cpu().tolist())
        elif dataset == "movielens1m":
            candidate_label = data[9].to(device)
            candidate_list.append(data[5].detach().cpu().tolist())
        else:
            raise ValueError("pls check the dataset name!")

        user_list.append(uid.tolist())
        candidate_label_list.append(candidate_label.detach().cpu().tolist())

        # * get model inputs
        feature_dict = get_feature_dict(
            dataset_dict=dataset_dict,
            model_params=model_params,
            data=data,
            device=device
            )

        # * forward 
        logits, loss_this_batch = model(
            candidate_label=candidate_label,
            feature_dict=feature_dict,
            )

        # * prediction
        try:
            if len(logits.size()) == 2:
                if logits.size()[-1] == 2:  # if the output size is [batch_size,2]
                    click_prob = logits[:, 1].tolist()
                    softmax_list.extend(click_prob)
                    prediction = torch.where(logits[:, 1] >= 0.5, 1, 0)
                    prediction_list.extend(prediction.tolist())
                elif logits.size()[-1] == 1:  # if the output size is [batch_size,1]
                    softmax_list.extend(logits.tolist())
                    prediction = torch.where(logits >= 0.5, 1, 0)
                    prediction_list.extend(prediction.tolist())
            elif len(logits.size()) == 1:  # if the output size is [batch_size,]
                softmax_list.extend(logits.tolist())
                prediction = torch.where(logits >= 0.5, 1, 0)
                prediction_list.extend(prediction.tolist())
        except:
            raise ValueError("pls check the output shape!")

        # * backward
        if mode == 'train':
            loss_this_batch.backward()
            optimizer.step()
        else:
            pass

        # cal loss
        loss_this_batch = loss_this_batch.item()
        loss += loss_this_batch

        # print every 100 batch
        if (i + 1) % 100 == 0:
            print('epoch:%d \t batch:%d/%d \t loss:%0.4f' % (epoch, i + 1, len(dataloader), loss / (i + 1)))
        else:
            pass

        if stop_step is None:
            pass
        else:
            if (i + 1) > stop_step:
                break
            else:
                pass

        # start_time = datetime.now()
        # torch.cuda.empty_cache()

    # * cal metrics after 1 epoch
    user_list = flatten(user_list)
    candidate_label_list = flatten(candidate_label_list)
    prediction_list = flatten(prediction_list)
    candidate_list = flatten(candidate_list)

    eval_dict = cal_avg_metrics(
        user_id_list=user_list,
        candidate_label_list=candidate_label_list,
        softmax_list=softmax_list,
        prediction_list=prediction_list,
        candidate_list=candidate_list,
        )

    print(
        "epoch:%d \t batch:%d/%d \t loss:%0.4f\n"
        % (epoch, i + 1, len(dataloader), loss / (i + 1))
        )

    return model, optimizer, loss / (i + 1), eval_dict
