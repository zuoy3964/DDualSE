import numpy as np
import pandas as pd
import dill
from operator import itemgetter
from gensim.models import Word2Vec
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import (RandomSampler, SequentialSampler,
                                      WeightedRandomSampler)
from datetime import datetime
from transformers import BertTokenizer, BertModel
import re
from tqdm import tqdm

tqdm.pandas(desc='pandas bar')
import os


def read_original_data(path: str):
    """
    Read the original data from the path
    """
    # Read the original data
    path = path + r"/bookcrossing/dataset/Preprocessed_data.csv"
    df = pd.read_csv(path, sep=',', encoding='latin-1')
    df.drop(df.columns[0], axis=1, inplace=True)
    return df


class items_new_processor:
    def __init__(self, behaviors, word_dim, path):
        self.path = path
        self.behaviors = behaviors

    def get_items_new_original(self):
        items_new = self.behaviors[
            ["book_title", "book_author", "year_of_publication", "publisher", "Summary", "Language", "Category"]].copy()
        # drop duplicates
        items_new.drop_duplicates(inplace=True)
        # add padding
        items_new.loc[items_new.shape[0]] = ["", "", 0., "", "", "", ""]
        # reset index
        items_new.reset_index(drop=True, inplace=True)
        # add item_id
        items_new["item_id"] = items_new.index
        return items_new

    def process_category(self, items_new):
        items_new["Category"] = items_new["Category"].apply(lambda x: x[2:-2] if x is not None else "")
        items_new["categoryID"] = items_new["Category"].astype('category').cat.codes
        items_new["yearID"] = items_new["year_of_publication"].astype('category').cat.codes
        items_new["languageID"] = items_new["Language"].astype('category').cat.codes
        items_new["publisherID"] = items_new["publisher"].astype('category').cat.codes
        return items_new

    def get_word2vec_corpus(self, items_new):
        corpus = items_new[["book_title", "Summary"]]
        corpus = corpus.astype(str).to_numpy().flatten().tolist()
        corpus = [re.split(" |\n", x.lower()) for x in corpus]
        authors = items_new["book_author"].astype(str).unique().tolist()
        authors = [[x.lower()] for x in authors]
        corpus.extend(authors)
        return corpus

    def get_sentence_array(self, sentence, word2vec):
        sentence = sentence.lower().split()
        if len(sentence) == 0:
            return np.zeros(word2vec.vector_size)
        else:
            vec = np.zeros(word2vec.vector_size)
            for word in sentence:
                if word in word2vec.wv:
                    vec += word2vec.wv[word]
                else:
                    pass
            return vec / len(sentence)

    def get_bert(self, path: str):
        print("loading BertModel...")
        start_time = datetime.now()
        tz = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        print("succeed! cost {}s!".format((datetime.now() - start_time).seconds))
        bert_model = bert_model.cuda()
        return tz, bert_model

    def get_sentence_array(self, text, bert_model, tokenizer) -> dict:
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
            max_length=512,  # maximum length of a sentence
            truncation=True,  # Truncate the sentence to max_length
            padding=True,  # Pad the sentence to the maximum length
            return_attention_mask=True,  # Generate the attention mask
            return_tensors='pt',  # ask the function to return PyTorch tensors
            )

        outputs = bert_model(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda())
        vector_representation = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
        return vector_representation

    def get_items_new(self):
        print("\nstart to get items new...")
        start_time = datetime.now()
        items_new = self.get_items_new_original()
        print("get items new original succeed! time expand {}s!: ".format((datetime.now() - start_time).seconds))

        print("\nstart to process category...")
        start_time = datetime.now()
        items_new = self.process_category(items_new)
        print("process category succeed! time expand {}s!: ".format((datetime.now() - start_time).seconds))

        print("\nstart to get bert vec...")
        start_time = datetime.now()
        tokenizer, bert_model = self.get_bert(path=self.path)
        items_new["book_title"] = items_new["book_title"].fillna("")
        items_new["Summary"] = items_new["Summary"].fillna("")
        items_new["book_author"] = items_new["book_author"].fillna("")
        book_title_dict = dict(
            zip(
                items_new["book_title"],
                items_new["book_title"].progress_apply(lambda x: self.get_sentence_array(x, bert_model, tokenizer))
                )
            )
        summary_dict = dict(
            zip(
                items_new["Summary"],
                items_new["Summary"].progress_apply(lambda x: self.get_sentence_array(x, bert_model, tokenizer))
                )
            )
        book_author_dict = dict(
            zip(
                items_new["book_author"],
                items_new["book_author"].progress_apply(lambda x: self.get_sentence_array(x, bert_model, tokenizer))
                )
            )
        print("get bert vec succeed! time expand {}s!: ".format((datetime.now() - start_time).seconds))

        return items_new, book_title_dict, summary_dict, book_author_dict


class behaviors_new_processor:
    def __init__(self, behaviors, history_len, padding_id):
        self.history_len = history_len
        self.padding_id = padding_id
        self.behaviors = behaviors[["user_id", "book_id", "label"]]
        self.behaviors.reset_index(drop=True, inplace=True)

        tmp_col = ["history_seq" + str(i) for i in range(history_len)]
        cols = ["user_id", "book_id", "label"] + tmp_col
        self.behaviors_new = pd.DataFrame(columns=cols)

    def get_a_sample(self, a_user_group):
        user_id = a_user_group["user_id"].iloc[0]
        book_ids = a_user_group["book_id"]
        labels = a_user_group["label"]

        history_seq = []
        book_ids_1 = book_ids[labels == 1]
        if len(book_ids_1) >= self.history_len:
            history_seq = book_ids_1[:self.history_len].tolist()
        else:
            history_seq = book_ids_1.tolist()
            if len(history_seq) < self.history_len:
                history_seq.extend([self.padding_id] * (self.history_len - len(history_seq)))
        rest_book_ids = book_ids[book_ids.isin(history_seq) == False].tolist()
        rest_labels = labels[book_ids.isin(history_seq) == False].tolist()

        tmp = pd.DataFrame(
            columns=["user_id", "book_id", "label"] + ["history_seq" + str(i) for i in range(self.history_len)]
            )
        tmp["user_id"] = [user_id] * len(rest_book_ids)
        tmp["book_id"] = rest_book_ids
        tmp["label"] = rest_labels
        tmp[tmp.columns[3:]] = history_seq
        return tmp

    def get_behaviors_new(self):
        print("\nstart to get behaviors new...")
        start_time = datetime.now()
        behaviors_new = self.behaviors.groupby("user_id").progress_apply(lambda x: self.get_a_sample(x))
        print("get behaviors new succeed! time expand {}s!".format((datetime.now() - start_time).seconds))
        behaviors_new.reset_index(drop=True, inplace=True)
        ratio = len(behaviors_new[behaviors_new["label"] == 1]) / len(behaviors_new[behaviors_new["label"] == 0])
        weight_pos = 1 / ratio
        weight_neg = 1
        behaviors_new["weight"] = behaviors_new["label"].apply(lambda x: weight_pos if x == 1 else weight_neg)
        behaviors_new[['user_id', 'book_id', 'label']] = behaviors_new[['user_id', 'book_id', 'label']].astype('int32')
        return behaviors_new


def split_train_test(behaviors_new, train_ratio=0.8):
    train_data = behaviors_new.groupby('user_id').progress_apply(
        lambda x: x.head(int(len(x) * train_ratio))
        ).reset_index(drop=True)
    valid_test_data = behaviors_new.groupby('user_id').progress_apply(
        lambda x: x.tail(int(len(x) * (1 - train_ratio)))
        ).reset_index(drop=True)
    valid_data = valid_test_data.groupby('user_id').progress_apply(lambda x: x.head(int(len(x) * 0.5))).reset_index(
        drop=True
        )
    test_data = valid_test_data.groupby('user_id').progress_apply(lambda x: x.tail(int(len(x) * 0.5))).reset_index(
        drop=True
        )

    return train_data, valid_data, test_data


def get_tensor(vec):
    vec = np.array(vec)
    vec = torch.from_numpy(vec)
    return vec


class MyDataSet(Dataset):
    def __init__(self, behaviors_new, items_new, book_title_dict, summary_dict, book_author_dict, history_len):
        self.behaviors_new = behaviors_new
        self.behaviors_seq = behaviors_new[["history_seq" + str(i) for i in range(history_len)]].values
        self.items_new = items_new
        self.book_title_dict = book_title_dict
        self.summary_dict = summary_dict
        self.book_author_dict = book_author_dict
        self.history_len = history_len

    def __len__(self):
        return len(self.behaviors_new)

    def __getitem__(self, idx):
        user_id = self.behaviors_new["user_id"].iloc[idx]
        history_seq = self.behaviors_seq[idx].tolist()
        history_title = self.items_new["book_title"].iloc[history_seq].tolist()
        history_title = np.array(list(map(self.book_title_dict.get, history_title)))
        history_summary = self.items_new["Summary"].iloc[history_seq].tolist()
        history_summary = np.array(list(map(self.summary_dict.get, history_summary)))
        history_author = self.items_new["book_author"].iloc[history_seq].tolist()
        history_author = np.array(list(map(self.book_author_dict.get, history_author)))
        history_year = self.items_new["yearID"].iloc[history_seq].tolist()
        history_category = self.items_new["categoryID"].iloc[history_seq].tolist()

        candidate_id = self.behaviors_new["book_id"].iloc[idx]
        candidate_title = self.items_new["book_title"].iloc[candidate_id]
        candidate_title = self.book_title_dict[candidate_title]
        candidate_summary = self.items_new["Summary"].iloc[candidate_id]
        candidate_summary = self.summary_dict[candidate_summary]
        candidate_author = self.items_new["book_author"].iloc[candidate_id]
        candidate_author = self.book_author_dict[candidate_author]
        candidate_year = self.items_new["yearID"].iloc[candidate_id]
        candidate_category = self.items_new["categoryID"].iloc[candidate_id]
        candidate_label = self.behaviors_new["label"].iloc[idx]

        user_id = get_tensor(user_id)
        history_seq = get_tensor(history_seq)
        history_title = get_tensor(history_title)
        history_summary = get_tensor(history_summary)
        history_author = get_tensor(history_author)
        history_year = get_tensor(history_year)
        history_category = get_tensor(history_category)
        candidate_id = get_tensor(candidate_id)
        candidate_title = get_tensor(candidate_title)
        candidate_summary = get_tensor(candidate_summary)
        candidate_author = get_tensor(candidate_author)
        candidate_year = get_tensor(candidate_year)
        candidate_category = get_tensor(candidate_category)
        candidate_label = get_tensor(candidate_label)

        return user_id, history_seq, \
            history_title, history_summary, history_author, history_category, history_year, \
            candidate_id, candidate_title, candidate_summary, candidate_author, candidate_category, candidate_year, \
            candidate_label


def load_dill_with_progress(file_path):
    with open(file_path, 'rb') as f:
        file_size = os.path.getsize(file_path)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc='Loading Dill') as pbar:
            data = dill.load(f)
            pbar.update(file_size)
    return data


def get_data(path, history_len, batch_size, word_dim=768, num_workers=10):

    try:
        print("\nstart to load data from{}...".format(path + r"/bookcrossing/data_processing/"))
        start_time = datetime.now()

        behaviors_new = load_dill_with_progress(
            path + r"/bookcrossing/data_processing/behaviors_new_" + str(history_len) + ".pkl"
            )
        items_new = load_dill_with_progress(path + r"/bookcrossing/data_processing/items_new.pkl")
        book_title_dict = load_dill_with_progress(
            path + r"/bookcrossing/data_processing/book_title_dict_" + str(word_dim) + ".pkl"
            )
        summary_dict = load_dill_with_progress(
            path + r"/bookcrossing/data_processing/summary_dict_" + str(word_dim) + ".pkl"
            )
        book_author_dict = load_dill_with_progress(
            path + r"/bookcrossing/data_processing/book_author_dict_" + str(word_dim) + ".pkl"
            )

        print("load data succeed! time expand {}s!".format((datetime.now() - start_time).seconds))

    except:
        print("load data failed! start to process data...")
        behaviors = read_original_data(path)
        processor = items_new_processor(behaviors=behaviors, word_dim=word_dim, path=path)
        items_new, book_title_dict, summary_dict, book_author_dict = processor.get_items_new()
        book_title2id = dict(zip(items_new["book_title"], items_new["item_id"]))
        behaviors["book_id"] = behaviors["book_title"].apply(lambda x: book_title2id[x])
        behaviors["label"] = behaviors["rating"].apply(lambda x: 1 if x >= 5 else 0)
        behaviors_new = behaviors_new_processor(
            behaviors=behaviors, history_len=history_len, padding_id=items_new["item_id"].iloc[-1]
            ).get_behaviors_new()

        print("\nstart to save data...")
        with open(path + r"/bookcrossing/data_processing/behaviors_new_" + str(history_len) + ".pkl", "wb") as f:
            dill.dump(behaviors_new, f)
        with open(path + r"/bookcrossing/data_processing/items_new.pkl", "wb") as f:
            dill.dump(items_new, f)
        with open(path + r"/bookcrossing/data_processing/book_title_dict_" + str(word_dim) + ".pkl", "wb") as f:
            dill.dump(book_title_dict, f)
        with open(path + r"/bookcrossing/data_processing/summary_dict_" + str(word_dim) + ".pkl", "wb") as f:
            dill.dump(summary_dict, f)
        with open(path + r"/bookcrossing/data_processing/book_author_dict_" + str(word_dim) + ".pkl", "wb") as f:
            dill.dump(book_author_dict, f)
        print("save data succeed!")

    print("\nstart to split train/valid/test...")
    start_time = datetime.now()
    train_data, valid_data, test_data = split_train_test(behaviors_new, train_ratio=0.6)
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    print("split train/valid/test succeed! time expand {}s!".format((datetime.now() - start_time).seconds))
    print(
        "train_data: {:10,}\nvalid_data: {:10,}\ntest_data: {:10,}".format(
            len(train_data), len(valid_data), len(test_data)
            )
        )

    print("\nstart to get dataloader...")
    train_dataset = MyDataSet(
        behaviors_new=train_data,
        items_new=items_new,
        book_title_dict=book_title_dict,
        summary_dict=summary_dict,
        book_author_dict=book_author_dict,
        history_len=history_len
        )
    valid_dataset = MyDataSet(
        behaviors_new=valid_data,
        items_new=items_new,
        book_title_dict=book_title_dict,
        summary_dict=summary_dict,
        book_author_dict=book_author_dict,
        history_len=history_len
        )
    test_dataset = MyDataSet(
        behaviors_new=test_data,
        items_new=items_new,
        book_title_dict=book_title_dict,
        summary_dict=summary_dict,
        book_author_dict=book_author_dict,
        history_len=history_len
        )

    weighted_sampler = WeightedRandomSampler(
        train_dataset.behaviors_new["weight"].tolist(),
        len(train_dataset.behaviors_new)
        )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=weighted_sampler, num_workers=num_workers
        )
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("get dataloader succeed!")

    dic = {
        "train_loader": train_dataloader,
        "valid_loader": valid_dataloader,
        "test_loader": test_dataloader,
        "behaviors_new": behaviors_new,
        "items_new": items_new,
        }

    return dic
