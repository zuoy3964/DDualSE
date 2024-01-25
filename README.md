# DDualSE
This is the PyTorch project of the DDualSE-BERT base on Python 3.8.10.

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

The original data is stored in the ./dataset directory, and the download links have been provided in ./dataset/README.md.

The ./data_processing folder contains the data processing function `data_process.py`. The generated new files will also be stored in this folder.


### ./models
This folder contains the SE-BERT, DualSE-BERT and DDualSE-BERT model and the corresponding embedding file.


### ./run
This folder contains:
- `config.json`：Dataset and model configuration
- `utils.py`：Parameter parsing
- `get_feature_dict.py`: Input construction
- `metrics.py`: Evaluation
- `train_and_valid.py`: Runing pipeline for an epoch
- `main.py`: main file


## Run
### Pipeline
Enter the run folder, and run the `run.sh` script.
```shell
cd run
sh run.sh
```

You can run different datasets and model running parameters by modifying the parameters in `run.sh`. For example, if you want to run DDualSE-BERT on the MovieLens-1M dataset:
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

#### Dataset Parameters:
- `dataset_name`: Optional parameters are "mindlarge", "movielens1m", "bookcrossing"
- `history_len`: user historical interaction length
- `word_dim`: Word embedding dimension, which must be consistent with the word vector dimension of the BERT model used, the default is 768
- `feature_num`: Nmber of attributes. MovieLens-1M is default to 3, MINDLarge is default to 4, BookCrossing is default to 5
- `attack_method`: It would be activated when the dataset is MovieLens-1M. The following 5 enumerated values could be used: 'none', 'random', 'early', 'middle', 'late', the default is 'none'
- `attack_ratio`: It would be activated when the dataset is MovieLens-1M. The value range is 0 to 1, the default is 0.001


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
