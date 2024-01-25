from datetime import datetime

import gensim.downloader as api
import gensim.models.keyedvectors as word2vec
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

tqdm.pandas(desc='pandas bar')
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import dill
import os


# from transformers import BertTokenizer
# Initialization
# pandarallel.initialize(progress_bar=True)

def load_train(dataset_train_path: str) -> [pd.DataFrame, pd.DataFrame]:
    news = pd.read_csv(
        dataset_train_path + '/news.tsv',
        sep='\t',
        names=['news_id', 'category', 'subcategory', 'headline', 'abstract', 'url', 'headline_entity',
               'abstract_entity']
        )
    behaviors = pd.read_csv(
        dataset_train_path + '/behaviors.tsv',
        sep='\t',
        names=['impression_id', 'user_id', 'time', 'history_seq', 'behaviors_seq']
        )
    # behaviors.drop(['index'], axis=1, inplace=True)
    return news, behaviors


# * get history seq
def get_history_seq(behaviors: pd.DataFrame, history_len: int):
    pad_value = 'PAD'

    # split history seq and get the first 10 news
    behaviors['history_seq'] = behaviors['history_seq'].fillna('')
    behaviors['history_seq'] = behaviors['history_seq'].str.split()
    behaviors['history_seq'] = behaviors['history_seq'].apply(lambda x: x[:history_len])

    # padding
    behaviors['history_seq'] = behaviors['history_seq'].apply(lambda x: x + [pad_value] * (history_len - len(x)))

    # rename history seq cols to news1, news2, ..., news10
    news_columns = [f'history_news{i}' for i in range(1, history_len + 1)]
    behaviors[news_columns] = pd.DataFrame(behaviors['history_seq'].tolist(), index=behaviors.index)

    return behaviors


# * split user behaviors
def sample_split(behaviors: pd.DataFrame) -> pd.DataFrame:
    behaviors['behaviors_seq'] = behaviors['behaviors_seq'].str.split(' ')
    # split behaviors_seq to multiple rows
    behaviors = behaviors.explode('behaviors_seq')
    try:
        # if it is train data
        behaviors[['news_id', 'label']] = behaviors['behaviors_seq'].str.split('-', expand=True)
        behaviors['label'] = behaviors['label'].astype(int)
    except:
        # if it is test data
        behaviors["news_id"] = behaviors['behaviors_seq']
        behaviors['label'] = [0] * len(behaviors['news_id'])
    behaviors.reset_index(drop=True, inplace=True)
    return behaviors


def proceed_behaviors_new(behaviors_new, history_len: int):

    print("get history seq...")
    start_time = datetime.now()
    behaviors_new = get_history_seq(behaviors=behaviors_new, history_len=history_len)
    print("succeed! cost {}s!".format((datetime.now() - start_time).seconds))
    print("split user behaviors...")
    start_time = datetime.now()
    behaviors_new = sample_split(behaviors_new)
    print("succeed! cost {}s!".format((datetime.now() - start_time).seconds))
    behaviors_new['user_id'] = behaviors_new['user_id'].astype('category').cat.codes
    behaviors_new.reset_index(drop=True, inplace=True)
    print("add sample weight...")
    start_time = datetime.now()
    num_label_1 = behaviors_new[behaviors_new['label'] == 1].shape[0]
    num_label_0 = behaviors_new[behaviors_new['label'] == 0].shape[0]
    try:
        ratio = int(num_label_0 / num_label_1)
        print("pos:neg = {}:{}".format(1, ratio))
    except:
        ratio = 1
    behaviors_new['weight'] = behaviors_new['label'].apply(lambda x: ratio if x == 1 else 1)
    print("succeed! cost {}s!".format((datetime.now() - start_time).seconds))

    return behaviors_new


# * pad news
def add_pad_value(value):
    if isinstance(value, (list, np.ndarray)):
        return np.zeros(len(value))
    elif isinstance(value, str):
        return ""
    else:
        return value


# * proceed_news_and_behaviors
def proceed_news_and_behaviors(news_new: pd.DataFrame, behaviors_new: pd.DataFrame) \
        -> [pd.DataFrame, pd.DataFrame]:

    print("padding...")
    news_ = news_new.iloc[len(news_new) - 1].map(add_pad_value)
    news_new = news_new.append(news_, ignore_index=True)
    news_new.loc[len(news_new) - 1, "news_id"] = 'PAD'

    print("remap news_id...")
    start_time = datetime.now()
    unique_news_ids = news_new['news_id'].unique()
    id_mapping = {news_id: idx for idx, news_id in enumerate(unique_news_ids)}

    news_new['news_id'] = news_new['news_id'].map(id_mapping)
    news_columns = [col for col in behaviors_new.columns if 'news' in col]
    behaviors_new[news_columns] = behaviors_new[news_columns].applymap(lambda x: id_mapping.get(x, 0))
    print("succeed! cost {}s!".format((datetime.now() - start_time).seconds))

    # ! proceed category
    print("remap category...")
    news_new['category'] = news_new['category'].replace('', '0').astype('category').cat.codes
    news_new['subcategory'] = news_new['subcategory'].replace('', '0').astype('category').cat.codes

    print("proceed NAN value...")
    news_new['headline'].fillna('', inplace=True)
    news_new['abstract'].fillna('', inplace=True)

    return news_new, behaviors_new


# sentence to vector
def sentence_to_vector(sentence, pretrain_vectors):
    words = sentence.split()
    vector = np.zeros(300)
    count = 0
    for word in words:
        if word in pretrain_vectors:
            vector += pretrain_vectors[word]
            count += 1
    if count != 0:
        vector /= count
    return vector


def get_pretrain_vectors(model_path: str):
    try:
        print("load word2vec-google-news-300.gz...")
        pretrain_vectors = word2vec.KeyedVectors.load_word2vec_format(
            model_path + "/word2vec-google-news-300.gz",
            binary=True
            )
    except:
        print("failed! load model online...")
        pretrain_vectors = api.load("word2vec-google-news-300")
    return pretrain_vectors


def get_text_vec_dict(df: pd.DataFrame, pretrain_vectors) -> dict:
    """"
    Args:
        df: 2 cols, news_id and text
        pretrain_vectors: word vec
    Returns:
        dict:{news_id, text_vec}={int: np.array}
    """
    # get vec for each text
    text_vec_dict = dict.fromkeys(df["news_id"].unique())
    for news_id, text in tqdm.tqdm(df.values):
        if text == '':
            text_vec_dict[news_id] = np.zeros(300)
        else:
            text_vec_dict[news_id] = sentence_to_vector(
                text,
                pretrain_vectors
                )
    return text_vec_dict


def get_bert(path: str):
    print("loading BertModel...")
    start_time = datetime.now()
    tz = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    print("succeed! cost {}s!".format((datetime.now() - start_time).seconds))
    bert_model = bert_model.cuda()
    return tz, bert_model


def get_sentence_array(text, bert_model, tokenizer) -> dict:
    """"
    Args:
        news_id: news_id
        text: text
        tokenizer: bert tokenizer
    Returns:
        dict:{news_id, text_vec}={int: np.array}
    """
    # Encode the sentence
    inputs = tokenizer.encode_plus(
        text=text,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length=300,  # maximum length of a sentence
        pad_to_max_length=True,  # Add [PAD]s
        return_attention_mask=True,  # Generate the attention mask
        return_tensors='pt',  # ask the function to return PyTorch tensors
        )
    outputs = bert_model(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda())
    vector_representation = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
    return vector_representation


def get_text_dict(news_df: pd.DataFrame, path: str) -> (pd.DataFrame, dict, dict):

    # ! proceed text
    # pretrain_vectors = get_pretrain_vectors(model_path=model_path)
    tokenizer, bertmodel = get_bert(path=path)

    print("get headline dict...")
    # headline_dict = get_text_vec_dict(df=news_df[["news_id", "headline"]], pretrain_vectors=pretrain_vectors)
    headline_dict = dict(
        zip(
            news_df["headline"], news_df["headline"].progress_apply(
                lambda x: get_sentence_array(text=x, bert_model=bertmodel, tokenizer=tokenizer)
                )
            )
        )
    print("get abstract dict...")
    # abstract_dict = get_text_vec_dict(df=news_df[["news_id", "abstract"]], pretrain_vectors=pretrain_vectors)
    abstract_dict = dict(
        zip(
            news_df["abstract"], news_df["abstract"].progress_apply(
                lambda x: get_sentence_array(text=x, bert_model=bertmodel, tokenizer=tokenizer)
                )
            )
        )

    return headline_dict, abstract_dict


def get_train_valid_test_data(behaviors_new: pd.DataFrame):
    train_data, valid_data = train_test_split(
        behaviors_new,
        test_size=0.4,
        )

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)

    return train_data, valid_data


# rewrite Dataset
class NewsDataset(Dataset):
    def __init__(self, news_new: pd.DataFrame, behaviors_new: pd.DataFrame, headline_dict: dict, abstract_dict: dict):
        self.news_new = news_new
        self.behaviors_new = behaviors_new

        try:
            self.impression_id = self.behaviors_new['impression_id']
        except:
            self.impression_id = self.behaviors_new['user_id']

        self.user_id = self.behaviors_new['user_id']
        self.history_seq = self.behaviors_new[[col for col in self.behaviors_new.columns if 'history_news' in col]]
        self.news_id_category_map = self.news_new[['news_id', 'category']].to_numpy()
        self.news_id_subcategory_map = self.news_new[['news_id', 'subcategory']].to_numpy()

        self.news_id = self.behaviors_new['news_id']
        try:
            self.label = self.behaviors_new['label']
        except:
            self.label = pd.Series([0] * len(self.behaviors_new))
        try:
            self.weight = self.behaviors_new['weight']
        except:
            self.weight = pd.Series([0] * len(self.behaviors_new))

        self.headline_dict = headline_dict
        self.abstract_dict = abstract_dict

        self.word_dim = next(iter(self.headline_dict.values())).shape[0]

        self.headline_get = np.vectorize(lambda x: self.headline_dict[x])
        self.abstract_get = np.vectorize(lambda x: self.abstract_dict[x])

    def __len__(self):
        return self.behaviors_new.shape[0]

    def __getitem__(self, index):
        impression_id = self.impression_id.loc[index]
        user_id = self.user_id.loc[index]
        history_seq = self.history_seq.loc[index].values

        history_seq_category = self.news_id_category_map[
            np.searchsorted(self.news_id_category_map[:, 0], history_seq), 1]
        history_seq_subcategory = self.news_id_subcategory_map[
            np.searchsorted(self.news_id_subcategory_map[:, 0], history_seq), 1]

        history_seq_headline = self.news_new['headline'][history_seq].values
        history_seq_headline = np.array(
            list(map(lambda x: self.headline_dict.get(x, np.zeros(self.word_dim)), history_seq_headline))
            )

        history_seq_abstract = self.news_new['abstract'][history_seq].values
        history_seq_abstract = np.array(
            list(map(lambda x: self.abstract_dict.get(x, np.zeros(self.word_dim)), history_seq_abstract))
            )

        candidate_news = self.news_id.loc[index]
        candidate_news_category = self.news_new.loc[np.in1d(self.news_new['news_id'], candidate_news), 'category']
        candidate_news_subcategory = self.news_new.loc[np.in1d(self.news_new['news_id'], candidate_news), 'subcategory']

        candidate_news_headline = self.news_new['headline'][candidate_news]
        candidate_news_headline = np.array(self.headline_dict.get(candidate_news_headline, np.zeros(self.word_dim)), )
        candidate_news_abstract = self.news_new['abstract'][candidate_news]
        candidate_news_abstract = np.array(self.abstract_dict.get(candidate_news_abstract, np.zeros(self.word_dim)), )

        label = self.label.loc[index]
        weight = self.weight.loc[index]

        # To tensor
        user_id = torch.tensor(user_id)
        history_seq = torch.from_numpy(np.array(history_seq))
        history_seq_category = torch.from_numpy(np.array(history_seq_category))
        history_seq_subcategory = torch.from_numpy(np.array(history_seq_subcategory))
        history_seq_headline = torch.from_numpy(np.array(history_seq_headline))
        history_seq_abstract = torch.from_numpy(np.array(history_seq_abstract))
        candidate_news = torch.tensor(candidate_news)
        candidate_news_category = torch.from_numpy(np.array(candidate_news_category)).squeeze(0)
        candidate_news_subcategory = torch.from_numpy(np.array(candidate_news_subcategory)).squeeze(0)

        candidate_news_headline = torch.from_numpy(np.array(candidate_news_headline))
        candidate_news_abstract = torch.from_numpy(np.array(candidate_news_abstract))
        label = torch.tensor(label)
        weight = torch.tensor(weight)

        return user_id, history_seq_category, history_seq_subcategory, history_seq_headline, history_seq_abstract, \
            candidate_news, candidate_news_category, candidate_news_subcategory, candidate_news_headline, \
            candidate_news_abstract, label, weight, history_seq, impression_id


def get_sampler(dataset, sampler_type: str, train_data: pd.DataFrame):
    """
    desc: get sampler
    params:
        dataset: Dataset
        sampler_type: str, sampler type
        train_data: pd.DataFrame
    return:
        sampler: torch.utils.data.sampler, sampler
    """
    if sampler_type == 'weighted':
        weights = train_data["weight"].values
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            )
    elif sampler_type == 'random':
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    elif sampler_type == "sequential":
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    else:
        raise NotImplementedError(f"Sampler type {sampler_type} not implemented.")
    return sampler


def load_dill_with_progress(file_path):
    with open(file_path, 'rb') as f:
        file_size = os.path.getsize(file_path)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc='Loading Dill') as pbar:
            data = dill.load(f)
            pbar.update(file_size)
    return data


def get_data(path, history_len, batch_size, word_dim, num_workers):

    try:
        path = r"/home/monijia/Graduate"
        print("\nread proceeded news_new and behaviors_new...")
        start_time = datetime.now()
        news_new = load_dill_with_progress(path + "/mindlarge/data_processing/news_new_train.pkl")
        behaviors_new = load_dill_with_progress(
            path + "/mindlarge/data_processing/behaviors_new_train_" + str(history_len) + ".pkl"
            )
        print("read successfully! cost {}s!".format((datetime.now() - start_time).seconds))
    except:
        print("failed! reproceed data...")
        print("\nread original data...")
        start_time = datetime.now()
        data_path = path + "/mindlarge/dataset/MINDlarge_train"
        news, behaviors = load_train(dataset_train_path=data_path)
        print("read successfully! cost {}s!".format((datetime.now() - start_time).seconds))

        print("\nget behaviors_new...")
        start_time = datetime.now()
        behaviors_new = proceed_behaviors_new(behaviors_new=behaviors, history_len=history_len)
        print("succeed! cost {}s!".format((datetime.now() - start_time).seconds))

        print("\nproceed news and behaviors_new...")
        start_time = datetime.now()
        news_new, behaviors_new = proceed_news_and_behaviors(news_new=news, behaviors_new=behaviors_new)

        # save
        print("\nsave data...")
        news_new.to_pickle(path + "/mindlarge/data_processing/news_new_train.pkl")
        behaviors_new.to_pickle(path + "/mindlarge/data_processing/behaviors_new_train_" + str(history_len) + ".pkl")
        print("save successfully! cost {}s!".format((datetime.now() - start_time).seconds))

    try:
        print("\nread proceeded text vec dict...")
        start_time = datetime.now()
        headline_dict = load_dill_with_progress(
            path + "/mindlarge/data_processing/headline_dict_train_" + str(word_dim) + ".pkl"
            )
        abstract_dict = load_dill_with_progress(
            path + "/mindlarge/data_processing/abstract_dict_train_" + str(word_dim) + ".pkl"
            )
        print("read successfully! cost{}s!".format((datetime.now() - start_time).seconds))

    except:
        print("failed! reproceed data...")
        start_time = datetime.now()
        headline_dict, abstract_dict = get_text_dict(news_df=news_new, path=path)

        print("\nsave data...")
        dill.dump(
            headline_dict, open(path + "/mindlarge/data_processing/headline_dict_train_" + str(word_dim) + ".pkl", "wb")
            )
        dill.dump(
            abstract_dict, open(path + "/mindlarge/data_processing/abstract_dict_train_" + str(word_dim) + ".pkl", "wb")
            )
        print("save successfully!")

    print("\nsplit train and valid data...")
    start_time = datetime.now()
    train_data, valid_data = get_train_valid_test_data(behaviors_new=behaviors_new)
    print(
        "\ntrain_dataset_shape:{:,}\n" \
        "valid_dataset_shape:{:,}\n" \
        "test_dataset_shape:{:,}".format(
            len(train_data), len(valid_data), 0
            )
        )
    print("split successfully! cost {}s!".format((datetime.now() - start_time).seconds))

    print("\ncreate dataset...")
    start_time = datetime.now()
    train_dataset = NewsDataset(
        news_new=news_new,
        behaviors_new=train_data,
        headline_dict=headline_dict,
        abstract_dict=abstract_dict
        )

    valid_dataset = NewsDataset(
        news_new=news_new,
        behaviors_new=valid_data,
        headline_dict=headline_dict,
        abstract_dict=abstract_dict
        )

    # test_dataset = NewsDataset(
    #     news_new=news_new_test,
    #     behaviors_new=behaviors_new_test,
    #     headline_dict=headline_dict_test,
    #     abstract_dict=abstract_dict_test
    # )
    print("succeed! cost {}s!".format((datetime.now() - start_time).seconds))

    print("\nCreate DataLoader...")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        shuffle=True,
        )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        shuffle=True,
        )

    # test_dataloader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=batch_size,
    #     drop_last=False,
    #     num_workers=20,
    #     shuffle=False,
    # )

    data_dict = {
        # loaders
        "train_loader": train_dataloader,
        "valid_loader": valid_dataloader,
        # "test_loader": test_dataloader,
        # new_dataframes
        "items_new": news_new,
        "behaviors_new": behaviors_new,
        # "news_new_test": news_new_test,
        # "behaviors_new_test": behaviors_new_test,
        }


    return data_dict
