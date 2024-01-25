# DDualSE
This is the PyTorch project of the DDualSE-BERT base on Python 3.8.10.
>这是一个基于Python 3.8.10的DDualSE-BERT的PyTorch项目。

Paper: `DDualSE: Decoupled Dual-head Squeeze and Excitation Attention For Sequential Recommendation`


## Enviroment 
```
dill==0.3.4
gensim==4.2.0
numpy==1.21.5
pandarallel==1.6.5
pandas==1.3.5
scikit-learn==1.0.1
torch==1.10.1+cu113
tqdm==4.64.0
transformers==4.18.0
```

## Folders
### ./datasets
We provide 3 datasets, including MINDLarge, MovieLens-1M, and BookCrossing, stored in the folders mindlage, movielens1m, and bookcrossing, respectively. Each folder has 2 subfolders: ./dataset and ./data_processing.
> 我们提供了3个数据集, 包括 MINDLarge, MovieLens-1M 和 BookCrossing, 它们分别存储在 mindlage, movielens1m 和 bookcrossing文件夹中。每个文件夹有2个子文件夹: ./dataset 和 ./data_processing。


The raw data is stored in the ./dataset folder. The original data for MovieLens-1M and BookCrossing has already been placed in the dataset folder. However, due to the large size of the raw data for MINDLarge, it needs to be downloaded manually from the official website at: https://msnews.github.io/.
>./dataset 里存放的是原始数据, MovieLens-1M和BookCrossing的原始数据已经放到了dataset文件夹中。由于MINDLarge的原始数据太大，需要自行从官网下载，链接为：https://msnews.github.io/。


The ./data_processing folder contains the data processing function `data_process.py`. The generated new files will also be stored in this folder.
>./data_processing 里存放了数据处理函数 `data_process.py`。生成的新文件也会存放在此文件夹中。


### ./models
This folder contains the SE-BERT, DualSE-BERT and DDualSE-BERT model and the corresponding embedding file.
> 此文件夹存放了SE-BERT、DualSE-BERT和DDualSE-BERT模型以及嵌入文件。


### ./run
This folder contains:
- `config.json`：Dataset and model configuration
- `utils.py`：Parameter parsing
- `get_feature_dict.py`: Input construction
- `metrics.py`: Evaluation
- `train_and_valid.py`: Runing pipeline for an epoch
- `main.py`: main file

>该目录包括：
- `config.json`：数据集和模型配置
- `utils.py`：参数解析
- `get_feature_dict.py`: 输入构造函数
- `metrics.py`: 评估函数
- `train_and_valid.py`: 一个epoch的运行流程
- `main.py`: 主运行文件

## Run
### Pipeline
Enter the run folder, and run the `run.sh` script.
> 先进入`run`文件夹，再运行`run.sh`脚本。
```shell
cd run
sh run.sh
```

You can run different datasets and model running parameters by modifying the parameters in `run.sh`. For example, if you want to run DDualSE-BERT on the MovieLens-1M dataset:
>可以通过修改`run.sh` 中的参数来运行不同的数据集和模型运行参数。例如, 如果想在 MovieLens-1M 数据集上运行 DDualSE-BERT:
```shell
python3 main.py \
    --dataset movielens1m \
    --history_len 90 \
    --model DDualSEBERT \
    --ebd_method fine-grained \
    --num_layer 8 \
    --num_head 16 \
    --fusion_method gate \
    --if_checkpoint False\
```
### Parameters in `run.sh`
The following is a description of some of the parameters:
>以下是其中的一些参数说明：

#### Dataset Parameters:
- `dataset_name`: Optional parameters are "mindlarge", "movielens1m", "bookcrossing"
- `history_len`: user historical interaction length
- `word_dim`: Word embedding dimension, which must be consistent with the word vector dimension of the BERT model used, the default is 768
- `feature_num`: Nmber of attributes. MovieLens-1M is default to 3, MINDLarge is default to 4, BookCrossing is default to 5
- `attack_method`: It would be activated when the dataset is MovieLens-1M. The following 5 enumerated values could be used: 'none', 'random', 'early', 'middle', 'late', the default is 'none'
- `attack_ratio`: It would be activated when the dataset is MovieLens-1M. The value range is 0 to 1, the default is 0.001

>数据集参数：
- `dataset_name`：可选参数有 "mindlarge", "movielens1m", "bookcrossing"
- `history_len`：用户历史交互长度
- `word_dim`：词向量维度，必须与使用的BERT模型的词向量维度一致，默认为768
- `feature_num`：属性数量。MovieLens-1M默认为3，MINDLarge默认为4，BookCrossing默认为5
- `attack_method`：当数据集为MovieLens-1M时，会被激活。可以使用以下5个枚举值：'none', 'random', 'early', 'middle', 'late'，默认为'none'
- `attack_ratio`：当数据集为MovieLens-1M时，会被激活。值范围为0到1，默认为0.001

#### Model Parameters:
- `model_name`: We provide "SEBERT","DualSEBERT" and "DDualSEBERT" here
- `ebd_method`: Embedding method, we provide "fine-grained" and "coarse-grained" here, the default is "fine-grained"
- `model_dim`: Model dimension, the default is 64
- `num_layer`: Number of layers, the default is 8
- `num_head`: Number of heads, the default is 8
- `gpu`: GPU number, if you have multiple GPUs, you can specify the GPU number here, the default is 0
- `optimizer_name`: You can choose a optimizer which torch provides, the default is "AdamW"
- `loss_method`: We provide "crossentropy" and "focalloss" here, the default is "crossentropy"
- `fusion_method`: We provide "gate", "sum" and "mean" for DualSEBERT and DDualSEBERT, and "gate" and "mean" for SEBERT. The default is "gate"

>模型参数：
- `model_name`：我们提供了"SEBERT","DualSEBERT"和"DDualSEBERT"
- `ebd_method`：嵌入方法，我们提供了"fine-grained"和"coarse-grained"，默认为"fine-grained"
- `model_dim`：模型维度，默认为64
- `num_layer`：层数，默认为8
- `num_head`：头数，默认为8
- `gpu`：GPU编号，如果有多个GPU，可以在这里指定GPU编号，默认为0
- `optimizer_name`：可以选择torch提供的优化器，默认为"AdamW"
- `loss_method`：我们提供了"crossentropy"和"focalloss"，默认为"crossentropy"
- `fusion_method`：我们为DualSEBERT和DDualSEBERT提供了"gate"、"sum"和"mean"，为SEBERT提供了"gate"和"mean"，默认为"gate"


#### Training Parameters:
- `path`: The project path, the default is "./"
- `seed`: Random seed for GPU, the default is 2022
- `if_checkpoint`: Whether to save the checkpoint, the default is "False"
- `num_workers`: Number of CPU workers, the default is 1
- `batch_size`: The default is 128
- `stop_step`: Control the number of steps per epoch, values greater than this will advance to the next epoch for training, the default is 100000000
- `lr`: Learning rate, the default is 0.001
- `weight_decay`: The default is 0.0001
- `epoch_num`: Total number of epochs, the default is 50
- `from_checkpoint`: The existed checkpoint path, the default is "none"

>训练参数：
- `path`：项目路径，默认为"./"
- `seed`：GPU的随机种子，默认为2022
- `if_checkpoint`：是否保存checkpoint，默认为"False"
- `num_workers`：CPU worker数量，默认为1
- `batch_size`：默认为128
- `stop_step`：控制每个epoch的步数，大于该值的将进入下一个epoch进行训练，默认为100000000
- `lr`：学习率，默认为0.001
- `weight_decay`：默认为0.0001
- `epoch_num`：总epoch数，默认为50
- `from_checkpoint`：已存在的checkpoint路径，默认为"none"
