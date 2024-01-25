import os
from datetime import datetime
import random
import sys
import json
import numpy as np
import importlib
import pandas as pd
from pprint import pprint
import torch
import torch.nn as nn

sys.path.append(r'../')
sys.path.append(r'../datasets/')
# print system path
print("\n".join(sys.path))
from train_and_valid import *
from utils import *


# * set torch seed
def seed_torch(seed_num):
    random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(
        seed_num
        )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=3, sci_mode=False)
    return None


# * load data
def load_data(
        dataset: str, project_path: str, word_dim: int, history_len: int, batch_size: int, num_workers: int,
        attack_method: str, attack_ratio: float
        ):
    import_path = f'{dataset}.data_processing.data_process'
    data_getter = importlib.import_module(import_path)
    dataset_path = project_path + "/datasets/"

    if dataset == "movielens1m":
        _ = data_getter.get_data(
            path=dataset_path,
            history_len=history_len,
            word_dim=word_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            attack_method=attack_method,
            attack_ratio=attack_ratio,
            )
    else:
        _ = data_getter.get_data(
            path=dataset_path,
            history_len=history_len,
            word_dim=word_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            )

    train_loader = _['train_loader']
    valid_loader = _['valid_loader']
    # test_loader = _['test_loader']
    test_loader = None
    behaviors_new = _['behaviors_new']
    items_new = _['items_new']

    return train_loader, valid_loader, test_loader, behaviors_new, items_new


# * load model from python files
def load_model(model_name: str):
    import_path = 'models.' + model_name
    model = importlib.import_module(import_path)
    return model


# * get dataset params dict
def get_dataset_dict(dataset: str, items_new: pd.DataFrame, args):
    with open("config.json", 'r') as f:
        config_data = json.load(f)
    dataset_dict = config_data['dataset'][dataset]
    dataset_dict.update({"items_cnt": len(items_new) + 1})
    # if dataset_dict.keys in args, then update with args value
    for i in dataset_dict.keys():
        if hasattr(args, i):
            dataset_dict[i] = getattr(args, i)
        else:
            pass

    if dataset == "bookcrossing":
        dataset_dict.update(
            {
                "category_cnt": len(items_new["categoryID"].unique()) + 1,
                "year_cnt": len(items_new['yearID'].unique()) + 1,
                "year_cnt": len(items_new['yearID'].unique()) + 1,
                }
            )

    elif dataset in ('mindlarge', 'mindsmall'):
        dataset_dict.update(
            {
                "category_cnt": len(items_new["category"].unique()) + 1,
                "subcategory_cnt": len(items_new["subcategory"].unique()) + 1,
                }
            )

    elif dataset in ("movielens1m"):
        dataset_dict.update(
            {
                "category_cnt": len(items_new["category"].unique()) + 1,
                "year_cnt": len(items_new['year'].unique()) + 1,
                }
            )
    else:
        raise AttributeError('Only support bookcrossing/movielens1m/mindlarge!')

    print("\nDataset Dict:")
    pprint(dataset_dict)
    return dataset_dict


# * get model dict
def get_model_dict(dataset_dict: dict, model_name: str, args) -> dict:
    with open("config.json", 'r') as f:
        config_data = json.load(f)
    model_dict = config_data['model'][model_name]

    # if model_dict.keys in args, then update
    for i in model_dict.keys():
        if hasattr(args, i):
            model_dict[i] = getattr(args, i)
        else:
            pass

    model_dict.update({"history_len": dataset_dict["history_len"]})

    return model_dict


# * init model
def get_ebd_params(dataset_dict, ebd_method: str):
    dataset = dataset_dict['dataset_name']
    if ebd_method == "fine-grained":
        if dataset == 'bookcrossing':
            category_cnt = dataset_dict['category_cnt']
            year_cnt = dataset_dict['year_cnt']
            word_dim = dataset_dict['word_dim']
            ebd_params = [
                f"text_{word_dim}",
                f"text_{word_dim}",
                f"text_{word_dim}",
                f"categorial_{category_cnt}",
                f"categorial_{year_cnt}",
                ]
        elif dataset in ('mindlarge', 'mindsmall'):
            word_dim = dataset_dict['word_dim']
            category_cnt = dataset_dict['category_cnt']
            subcategory_cnt = dataset_dict['subcategory_cnt']
            ebd_params = [
                f"categorial_{category_cnt}",
                f"categorial_{subcategory_cnt}",
                f"text_{word_dim}",
                f"text_{word_dim}",
                ]
        elif dataset == "movielens1m":
            word_dim = dataset_dict['word_dim']
            category_cnt = dataset_dict['category_cnt']
            year_cnt = dataset_dict['year_cnt']
            ebd_params = [
                f"text_{word_dim}",
                f"categorial_{category_cnt}",
                f"categorial_{year_cnt}",
                ]
    elif ebd_method == "coarse-grained":
        items_cnt = dataset_dict['items_cnt']
        ebd_params = [
            f"sequential_{items_cnt}",
            ]
    elif ebd_method == "fine&coarse":
        if dataset == 'bookcrossing':
            category_cnt = dataset_dict['category_cnt']
            year_cnt = dataset_dict['year_cnt']
            word_dim = dataset_dict['word_dim']
            items_cnt = dataset_dict['items_cnt']
            ebd_params = [
                f"text_{word_dim}",
                f"text_{word_dim}",
                f"text_{word_dim}",
                f"categorial_{category_cnt}",
                f"categorial_{year_cnt}",
                f"sequential_{items_cnt}",
                ]
        elif dataset in ('mindlarge', 'mindsmall'):
            word_dim = dataset_dict['word_dim']
            category_cnt = dataset_dict['category_cnt']
            subcategory_cnt = dataset_dict['subcategory_cnt']
            items_cnt = dataset_dict['items_cnt']
            ebd_params = [
                f"categorial_{category_cnt}",
                f"categorial_{subcategory_cnt}",
                f"text_{word_dim}",
                f"text_{word_dim}",
                f"sequential_{items_cnt}",
                ]
        elif dataset == "movielens1m":
            word_dim = dataset_dict['word_dim']
            category_cnt = dataset_dict['category_cnt']
            year_cnt = dataset_dict['year_cnt']
            items_cnt = dataset_dict['items_cnt']
            ebd_params = [
                f"text_{word_dim}",
                f"categorial_{category_cnt}",
                f"categorial_{year_cnt}",
                f"sequential_{items_cnt}",
                ]
    return ebd_params


def print_args(data_params, model_params, args, args_ms):

    print('\n*********Data Processing Parameters*********')
    for attr, value in sorted(data_params.items()):
        print('{}={}'.format(attr.upper(), value))
        args_ms_new = "{0}={1}\n".format(attr, str(value))
        args_ms += args_ms_new

    args_ms += "*********Model Parameters*********\n"
    print('*********Model Parameters*********')
    for attr, value in sorted(model_params.items()):
        print('{}={}'.format(attr.upper(), value))
        args_ms_new = "{0}={1}\n".format(attr, str(value))
        args_ms += args_ms_new

    args_ms += "*********Training Parameters*********\n"
    print("*********Training Parameters*********")
    for attr, value in sorted(args.__dict__.items()):
        if attr in ('batch_size', 'lr', 'weight_decay', 'optimizer_name', 'epoch_num', 'from_checkpoint'):
            print('{}={}'.format(attr.upper(), value))
            args_ms_new = "{0}={1}\n".format(attr, str(getattr(args, attr)))
            args_ms += args_ms_new

    args_ms += "*********Other Parameters*********\n"
    print('*********Other Parameters*********')
    for attr, value in sorted(args.__dict__.items()):
        if attr in ('if_checkpoint', 'seed', 'send_email', 'if_get_writer', 'stop_step'):
            print('{}={}'.format(attr.upper(), value))
            args_ms_new = "{0}={1}\n".format(attr, str(getattr(args, attr)))
            args_ms += args_ms_new

    return args_ms


if __name__ == '__main__':

    args = parseArgs()

    project_path = args.path
    dataset = args.dataset_name
    history_len = args.history_len
    word_dim = args.word_dim
    batch_size = args.batch_size
    if_checkpoint = args.if_checkpoint
    seed = args.seed
    gpu = args.gpu
    model_dim = args.model_dim
    model_name = args.model_name
    from_checkpoint = args.from_checkpoint
    weight_decay = args.weight_decay
    optimizer_name = args.optimizer_name
    epoch_num = args.epoch_num
    lr = args.lr
    stop_step = args.stop_step
    seed_torch(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda:" + str(args.gpu) if str(args.gpu).lower() != 'none' else "cpu")

    # ! get data
    train_loader, valid_loader, test_loader, behaviors_new, items_new = load_data(
        dataset=dataset,
        project_path=project_path,
        history_len=history_len,
        word_dim=word_dim,
        batch_size=batch_size,
        num_workers=args.num_workers,
        attack_method=args.attack_method if dataset == "movielens1m" else None,
        attack_ratio=args.attack_ratio if dataset == "movielens1m" else None,
        )


    # * Dataset Dict
    dataset_dict = get_dataset_dict(
        dataset=dataset,
        items_new=items_new,
        args=args,
        )

    # * Model Dict
    model_dict = get_model_dict(
        dataset_dict=dataset_dict,
        model_name=model_name,
        args=args,
        )

    # * init model
    module_py = load_model(
        model_name=model_name
        )
    ebd_params = get_ebd_params(
        dataset_dict=dataset_dict,
        ebd_method=model_dict['ebd_method'],
        )
    class_obj = getattr(
        module_py,
        model_name
        )
    model = class_obj(
        model_dim=model_dim,
        ebd_params=ebd_params,
        model_params=model_dict,
        )
    model = model.to(device)

    # * orthogonal init for DDualSEBERT
    if 'DDualSEBERT' in model_name:
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Embedding, nn.Parameter)):
                nn.init.orthogonal_(m.weight)

    # * print model params
    TotalParams = sum(p.numel() for p in model.parameters())
    ExpEbdParams = 0
    ebd_method = model_dict['ebd_method']
    print("\nLayer Params:")
    for name, param in model.named_parameters():
        print("{:20}\t{:10,}".format(name, param.numel()))
        if (ebd_method == "coarse-grained") and ("embed" in name.lower()):
            pass
        else:
            ExpEbdParams += param.numel()
    print("\nTotal Params：{:10,}".format(TotalParams))
    if ebd_method == "coarse-grained":
        print("ExpEbd Params：{:10,}".format(ExpEbdParams))
    model_dict.update({"TotalParams": TotalParams, "ExpEbdParams": ExpEbdParams})

    # * print args
    args_ms = "*********DataSet Params*********\n"
    ms = ""
    args_ms = print_args(
        data_params=dataset_dict,
        model_params=model_dict,
        args=args,
        args_ms=args_ms,
        )

    # * get optimizer
    try:
        optimizer = eval(
            "torch.optim." + optimizer_name
            )(
            model.parameters(),
            weight_decay=weight_decay, lr=lr
            )
    except AttributeError:
        print("optimizer only includes AdamW, Adam, SGD and so on...")

    # * train and valid
    startime = datetime.now().strftime("%Y%m%d_%H%M%S")

    valid_eval_best = {"AUC": 0, "nDCG@5": 0, "nDCG@10": 0, "MRR": 0}
    valid_epoch_best = 0
    epoch_start = 0

    # * load model from checkpoint
    if from_checkpoint is not None:
        with open(from_checkpoint + "/model.pt", 'rb') as f:
            model = torch.load(f)
        with open(from_checkpoint + "/optimizer.pt", 'rb') as f:
            optimizer = torch.load(f)
        epoch_start = eval(from_checkpoint[-1]) + 1
        print("\nload model from checkpoint succeed, path:", from_checkpoint)
    else:
        pass

    # * create checkpoint fold
    checkpoint_path = f'../checkpoint/{dataset}/{model_name}/{startime}'
    if if_checkpoint:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            print("\ncreate checkpoint to ", checkpoint_path)
        else:
            pass
        # create config.json
        with open(checkpoint_path + "/config.json", "w") as f:
            json.dump(args.__dict__, f, indent=4)
    else:
        pass

    # * train and valid
    for epoch in range(epoch_start, epoch_num):

        start_time = datetime.now()
        print(f"\n======================EPOCH={epoch}========================")
        print("*************TRAIN*************")
        model, optimizer, train_loss, train_eval_dict = train_or_valid(
            dataset_dict=dataset_dict,
            model_params=model_dict,
            model=model,
            mode='train',
            epoch=epoch,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            stop_step=stop_step,
            )

        print(f"1 TRAIN epoch cost {(datetime.now() - start_time).seconds} seconds\n")

        for key, value in train_eval_dict.items():
            if isinstance(value, float):
                value = format(value, '.4f')
            print(f'{key}:{value}')

        start_time = datetime.now()
        print("\n*************VALID*************")

        model, optimizer, valid_loss, valid_eval_dict = train_or_valid(
            dataset_dict=dataset_dict,
            model_params=model_dict,
            model=model,
            mode='valid',
            epoch=epoch,
            dataloader=valid_loader,
            optimizer=optimizer,
            device=device,
            stop_step=stop_step,
            )

        print(f"1 VALID epoch cost {(datetime.now() - start_time).seconds} seconds\n")

        for key, value in valid_eval_dict.items():
            if isinstance(value, float):
                value = format(value, '.4f')
            print(f'{key}:{value}')

        # update valid_auc_best
        valid_auc = valid_eval_dict['AUC']
        valid_auc_best = valid_eval_best['AUC']
        if valid_auc >= valid_auc_best:
            valid_eval_best = valid_eval_dict
            valid_epoch_best = epoch
            valid_eval_best.update({"epoch": f"{valid_epoch_best}/{epoch}"})
        else:
            valid_eval_best.update({"epoch": f"{valid_epoch_best}/{epoch}"})

        print("\n***********Best Result***********")
        for key, value in valid_eval_best.items():
            if isinstance(value, float):
                value = format(value, '.4f')
            print(f'{key}:{value}')

        # * save checkpoint
        if if_checkpoint:
            print("saving check point...")
            checkpoint_path_epoch = checkpoint_path + f'/epoch_{epoch}'
            if not os.path.exists(checkpoint_path_epoch):
                os.makedirs(checkpoint_path_epoch)
            else:
                pass
            torch.save(model, checkpoint_path_epoch + "/model.pt")
            torch.save(model.state_dict(), checkpoint_path_epoch + "/model_state_dict.pth")
            torch.save(optimizer, checkpoint_path_epoch + "/optimizer.pt")
            torch.save(optimizer.state_dict(), checkpoint_path_epoch + "/optimizer_state_dict.pth")
            result = {
                "model_name": model_name,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_eval_best": valid_eval_best,
                }
            with open(checkpoint_path_epoch + "/result.json", "w") as f:
                json.dump(result, f, indent=4)
            print("save checkpoint succeed!")
        else:
            pass

    print("\n***********Args******************\n", args_ms)
