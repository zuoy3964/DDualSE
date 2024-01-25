import argparse
import json


def read_config(config_path:str, dataset:str, model:str)->dict:
    with open(config_path, 'r') as f:
        config_data = json.load(f)
        dataset_config = config_data['dataset'][dataset]
        model_config = config_data['model'][model]
    return dataset_config, model_config


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parseArgs():

    parser = argparse.ArgumentParser(description='Parse config from JSON file')

    parser.add_argument('--path', type=str, default='../')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)

    dataset_config, model_config = read_config(
        config_path="config.json",
        dataset=parser.parse_known_args()[0].dataset,
        model=parser.parse_known_args()[0].model
        )

    # dataset config
    for key, value in dataset_config.items():
        parser.add_argument('--' + key, type=type(value), default=value)
    
    # model config
    for key, value in model_config.items():
        parser.add_argument('--' + key, type=type(value), default=value)

    # other config
    parser.add_argument('--seed', type=int, default=2022, nargs='?')
    parser.add_argument('--if_checkpoint', type=str2bool, nargs='?', default="Fasle")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--stop_step', type=int, default=100000000)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001, nargs='?')
    parser.add_argument('--epoch_num', type=int, default=50)
    parser.add_argument('--from_checkpoint', type=str, nargs='?', default=None, help='checkpoint path')

    args = parser.parse_args()

    return args
