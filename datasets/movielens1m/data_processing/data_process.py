import numpy as np
import pandas as pd
import dill
import os
import random
from operator import itemgetter
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import BertTokenizer, BertModel
from datetime import datetime
from tqdm import tqdm

tqdm.pandas(desc='pandas bar')


class read_original_data:
    def __init__(self, path):
        self.path = path
        print('read data from {}'.format(path))

    def get_behaviors(self):
        behaviors = pd.read_csv(
            self.path + '/ratings.dat',
            sep='::',
            header=None,
            engine='python'
            )
        behaviors.columns = ['uid', 'mid', 'rating', 'timestamp']
        return behaviors

    def get_movie(self):
        movie = pd.read_csv(
            self.path + '/movies.dat',
            sep='::',
            header=None,
            encoding='ISO-8859-1',
            engine='python'
            )
        movie.columns = ['mid', 'title', 'categories']
        movie = movie.fillna("")
        return movie


class processing_movie:
    def __init__(self, movie, path: str, word_dim=64):
        self.movie = movie
        self.path = path
        self.word_dim = word_dim
        print('processing movie data...')

    def padding(self):
        pad_idx = self.movie.shape[0]
        self.movie.loc[pad_idx] = [pad_idx, '', '']
        return self.movie

    def get_movie_categories(self, category_str):
        # get the first category
        categories = category_str.split('|')
        return categories[0]

    def code_category(self):
        # if null, code as 0
        self.movie['category'] = self.movie['categories'].apply(self.get_movie_categories)
        self.movie['categoryID'] = pd.Categorical(self.movie['category'])
        self.movie['categoryID'] = self.movie['categoryID'].cat.codes
        return self.movie

    def get_movie_year(self, title):
        year = title[-5:-1]
        return year

    def code_year(self):
        self.movie['year'] = self.movie['title'].apply(self.get_movie_year)
        self.movie['yearID'] = pd.Categorical(self.movie['year'])
        # if null, code as 0
        self.movie['yearID'] = self.movie['yearID'].cat.codes
        return self.movie

    def get_movie_title(self, title):
        title = title[:-7]
        return title

    def get_title(self):
        self.movie['title'] = self.movie['title'].apply(self.get_movie_title)
        return self.movie

    def get_movie(self):
        self.movie = self.padding()
        self.movie = self.code_category()
        self.movie = self.code_year()
        self.movie = self.get_title()
        self.movie = self.movie.reset_index(drop=True)
        return self.movie

    # def word2vec(self, corpus, window=5, min_count=1, workers=4):
    #     print('training word2vec...')
    #     word2vec = Word2Vec(corpus, vector_size=self.word_dim, window=window, min_count=min_count, workers=workers)
    #     print('training word2vec done!')
    #     return word2vec
    #
    # def get_sentence_array(self, title, word2vec):
    #     title = title.lower().split()
    #     if len(title) == 0:
    #         return np.zeros(word2vec.vector_size)
    #     else:
    #         vec = np.zeros(word2vec.vector_size)
    #         for word in title:
    #             try:
    #                 vec += word2vec.wv[word]
    #             except:
    #                 pass
    #         return vec / len(title)
    #
    # def get_movie_title_dict(self):
    #     movie_title_dict = {}
    #     movie_title_array = self.movie['title'].values
    #     corpus = [title.lower().split() for title in movie_title_array]
    #     word2vec = self.word2vec(corpus)
    #     for title in movie_title_array:
    #         movie_title_dict[title] = self.get_sentence_array(title, word2vec)
    #     return movie_title_dict

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
            text: text
            bert_model: bert model
            tokenizer: bert tokenizer
        Returns:
            dict:{text, text_vec}={int: np.array}
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

    def get_movie_title_dict(self):

        tokenizer, bert_model = self.get_bert(path=self.path)
        movie_title_dict = dict(
            zip(
                self.movie['title'],
                self.movie['title'].progress_apply(lambda x: self.get_sentence_array(x, bert_model, tokenizer))
                )
            )
        return movie_title_dict


# reindex movie id and remap
def map_new_id(movie, behaviors):
    movie_id2idx = {mid: idx for idx, mid in enumerate(movie['mid'])}
    movie['mid'] = movie['mid'].apply(lambda x: movie_id2idx[x])
    behaviors['mid'] = behaviors['mid'].apply(lambda x: movie_id2idx[x])
    return movie, behaviors


# choose the first 50 movies as history click for each user
def get_samples(behaviors_a_person, movie, history_len=50):
    # sorted by timestamp
    uid = behaviors_a_person['uid'].values[0]
    behaviors_a_person = behaviors_a_person.sort_values(by='timestamp', ascending=True)
    behaviors_a_person_pos = behaviors_a_person[behaviors_a_person['rating'] > 3]
    # get the history clicks
    behaviors_a_person_history = behaviors_a_person_pos["mid"].head(history_len).values
    if len(behaviors_a_person_history) < history_len:
        # pad
        pad_idx = movie.shape[0] - 1
        pad_len = history_len - len(behaviors_a_person_history)
        behaviors_a_person_history = np.concatenate([behaviors_a_person_history, pad_idx * np.ones(pad_len)])
        # print(len(behaviors_a_person_history))
    else:
        pass

    # choose the rest as candidate
    behaviors_a_person_pos = behaviors_a_person_pos[~behaviors_a_person_pos['mid'].isin(behaviors_a_person_history)]
    # pos sample
    behaviors_a_person_pos['label'] = [1] * len(behaviors_a_person_pos)

    # neg sample
    behaviors_a_person_neg = behaviors_a_person[behaviors_a_person['rating'] <= 3].copy()
    behaviors_a_person_neg['label'] = [0] * len(behaviors_a_person_neg)

    # concat pos and neg
    behaviors_a_person = pd.concat([behaviors_a_person_pos, behaviors_a_person_neg])

    # join behaviors_a_person_history
    behaviors_a_person_history = pd.DataFrame(behaviors_a_person_history).T.astype(int)
    behaviors_a_person_history.columns = ['history' + str(i) for i in range(history_len)]

    # expand behaviors_a_person_history
    if len(behaviors_a_person) > 0:
        behaviors_a_person_history = pd.concat(
            [behaviors_a_person_history] * len(behaviors_a_person), ignore_index=True
            )
    else:
        return None

    # reindex
    behaviors_a_person_history = behaviors_a_person_history.reset_index(drop=True)
    behaviors_a_person = behaviors_a_person.reset_index(drop=True)
    behaviors_a_person = pd.merge(
        behaviors_a_person, behaviors_a_person_history, how='left', left_index=True, right_index=True
        )

    return behaviors_a_person


# * Add Random Attack
def get_samples_random(behaviors_new, movie, attack_ratio=0.01):
    black_uid_cnt = int(len(behaviors_new['uid'].unique()) * attack_ratio)
    black_uid = np.arange(len(behaviors_new['uid'].unique()), len(behaviors_new['uid'].unique()) + black_uid_cnt)
    print('black_uid_cnt:{}'.format(black_uid_cnt))

    history_cols = [i for i in behaviors_new.columns if 'history' in i]
    history_len = len(history_cols)
    candidate_cnt = 20
    print("generate {} samples for each black uid!".format(candidate_cnt))

    for uid in black_uid:
        behaviors_a_person = pd.DataFrame(columns=behaviors_new.columns)
        uid_list = [uid] * candidate_cnt
        behaviors_a_person['uid'] = uid_list

        # random choose history seq
        history_seq = np.random.choice(movie['mid'].values, history_len)
        behaviors_a_person[history_cols] = history_seq

        # random choose candidates
        candidates = movie[~movie['mid'].isin(history_seq)]['mid'].values
        candidate_mid = np.random.choice(candidates, candidate_cnt)
        behaviors_a_person['mid'] = candidate_mid

        # random choose label
        label = np.random.choice([0, 1], candidate_cnt)
        behaviors_a_person['label'] = label

        # add to behaviors_new
        behaviors_a_person = behaviors_a_person.fillna(0)
        behaviors_new = pd.concat([behaviors_new, behaviors_a_person])

    return behaviors_new


# * Add Early Attack
def get_samples_early(behaviors_new, movie, attack_ratio=0.01):
    """Ramdom Change the behaviors of candidate"""
    num_noise_sample = int(len(behaviors_new) * attack_ratio)
    print("replace candidate of {} uids!".format(num_noise_sample))
    # choose the replaced samples
    random_indices = np.random.choice(behaviors_new.index, size=num_noise_sample)
    # choose the replaced movie id
    random_movie = movie['mid'].sample(num_noise_sample).values
    # replace
    behaviors_new.loc[random_indices, 'mid'] = random_movie

    return behaviors_new


# * Add Middle Attack
def get_samples_middle(behaviors_new, movie, attack_ratio=0.01):
    """Ramdom Change the history seq"""
    num_noise_sample = int(len(behaviors_new) * attack_ratio)
    print("replace history seq of {} uids!".format(num_noise_sample))
    # random choose the black uid group
    black_uids = np.random.choice(behaviors_new.index, size=num_noise_sample)
    # for each black uid, random change the history seq with the probability 0.5
    history_cols = [i for i in behaviors_new.columns if 'history' in i]
    behaviors_new.loc[black_uids, history_cols] = behaviors_new.loc[black_uids, history_cols].applymap(
        lambda x: movie['mid'].sample(1).values[0] if random.random() < 0.5 else x
        )

    return behaviors_new


# * Add Late Attack
def get_samples_late(behaviors_new, movie, attack_ratio=0.01):

    num_noise_sample = int(len(behaviors_new) * attack_ratio)
    print("replace history seq or candidate of {} uids!".format(num_noise_sample))
    # random choose the black uid group
    black_uids = np.random.choice(behaviors_new.index, size=num_noise_sample)

    # random choose the replaced movie_id cols
    history_cols = [i for i in behaviors_new.columns if 'history' in i]
    history_cols.append('mid')

    # for each black uid, random change the history seq with the probability 0.5
    behaviors_new.loc[black_uids, history_cols] = behaviors_new.loc[black_uids, history_cols].applymap(
        lambda x: movie['mid'].sample(1).values[0] if random.random() < 0.5 else x
        )

    return behaviors_new


# split train ,valid and test
def split_train_test(behaviors_new, train_ratio=0.8):

    behaviors_new = behaviors_new.sort_values(by='timestamp', ascending=True)
    # for a person, select the first 80% as train, the last 10% as valid, the last 10% as test
    train_data = behaviors_new.groupby('uid').apply(lambda x: x.head(int(len(x) * train_ratio))).reset_index(drop=True)
    valid_test_data = behaviors_new.groupby('uid').apply(lambda x: x.tail(int(len(x) * (1 - train_ratio)))).reset_index(
        drop=True
        )
    valid_data = valid_test_data.groupby('uid').apply(lambda x: x.head(int(len(x) * 0.5))).reset_index(drop=True)
    test_data = valid_test_data.groupby('uid').apply(lambda x: x.tail(int(len(x) * 0.5))).reset_index(drop=True)

    # add weights
    pos_cnt = len(train_data[train_data['label'] == 1])
    neg_cnt = len(train_data[train_data['label'] == 0])
    pos_weight = pos_cnt / neg_cnt
    neg_weight = 1.0
    train_data['weight'] = train_data['label'].apply(lambda x: pos_weight if x == 1 else neg_weight)

    return train_data, valid_data, test_data


class MyDataSet(Dataset):
    def __init__(self, behaviors, movie, movie_title_dict, history_len):
        self.behaviors = behaviors
        self.movie = movie
        self.uid = self.behaviors['uid'].values
        self.labels = self.behaviors['label'].values
        self.history = self.behaviors[['history' + str(i) for i in range(history_len)]].values
        self.mid = self.behaviors['mid'].values
        self.movie_title_dict = movie_title_dict

    def __getitem__(self, index):
        uid = self.uid[index]

        history_seq = self.history[index]
        # 同时取字典的多个值，用itemgetter
        history_title = self.movie['title'][history_seq].values
        getter = itemgetter(*history_title)
        history_title = np.array(getter(self.movie_title_dict))
        history_categoryID = self.movie['categoryID'][history_seq].values
        history_yearID = self.movie['yearID'][history_seq].values

        candidate_mid = self.mid[index]
        candidate_title = self.movie_title_dict[self.movie['title'][candidate_mid]]
        candidate_categoryID = self.movie['categoryID'][candidate_mid]
        candidate_yearID = self.movie['yearID'][candidate_mid]

        candidate_label = self.labels[index].astype(int)

        return uid, history_seq, history_title, history_categoryID, history_yearID, candidate_mid, candidate_title, candidate_categoryID, candidate_yearID, candidate_label

    def __len__(self):
        return len(self.behaviors)


def load_dill_with_progress(file_path):
    with open(file_path, 'rb') as f:
        file_size = os.path.getsize(file_path)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc='Loading Dill') as pbar:
            data = dill.load(f)
            pbar.update(file_size)
    return data


def get_data(
        path, history_len, batch_size, word_dim=768, num_workers=10, attack_method: str = "none", attack_ratio=0.01
        ):
    try:
        print('\nload behaviors_new & movie_new from {}...'.format(path))
        behaviors_new = load_dill_with_progress(
            path + '/movielens1m/data_processing/behaviors_new_' + str(history_len) + '.pkl'
            )
        movie_new = load_dill_with_progress(path + '/movielens1m/data_processing/movie_new.pkl')
        print('load data successfully!')

    except:

        print('\nload data failed! creating new data...')

        # read original data
        print('\nreading original data from {}...'.format(path))
        read_data = read_original_data(path + '/movielens1m/dataset')
        behaviors = read_data.get_behaviors()

        # processing movie data
        print('\nproceeding movie data...')
        movie = read_data.get_movie()
        movie_processor = processing_movie(movie=movie, path=path, word_dim=word_dim)
        movie = movie_processor.get_movie()

        # remap movie id
        print('\nremapping movie id...')
        movie_new, behaviors = map_new_id(movie, behaviors)

        # create behaviors_new
        print('\ncreating behaviors_new...')
        behaviors_new = behaviors.groupby('uid').apply(lambda x: get_samples(x, movie_new, history_len=history_len))
        behaviors_new.reset_index(drop=True, inplace=True)

        # save
        print('\nsave movie_new & behaviors_new to {}...'.format(path))
        dill.dump(movie_new, open(path + '/movielens1m/data_processing/movie_new.pkl', 'wb'))
        dill.dump(
            behaviors_new, open(path + '/movielens1m/data_processing/behaviors_new_' + str(history_len) + '.pkl', 'wb')
            )
        print('save data successfully!')

    np.random.seed(2022)
    if attack_method == "none":
        pass
    elif attack_method == "random":
        print('\nrandomly attack...')
        behaviors_new = get_samples_random(
            behaviors_new=behaviors_new,
            movie=movie_new,
            attack_ratio=attack_ratio
            )
        print('Attack Done!')
    elif attack_method == "early":
        print('\nearly attack...')
        behaviors_new = get_samples_early(
            behaviors_new=behaviors_new,
            movie=movie_new,
            attack_ratio=attack_ratio
            )
        print('Attack Done!')
    elif attack_method == "middle":
        print('\nmiddle attack...')
        behaviors_new = get_samples_middle(
            behaviors_new=behaviors_new,
            movie=movie_new,
            attack_ratio=attack_ratio
            )
        print('Attack Done!')
    elif attack_method == "late":
        print('\nlate attack...')
        behaviors_new = get_samples_late(
            behaviors_new=behaviors_new,
            movie=movie_new,
            attack_ratio=attack_ratio
            )
        print('Attack Done!')
    else:
        raise ValueError("attack_method should be 'none','random','early','middle' or 'late'!")

    try:
        print('\nload movie_title_dict from {}...'.format(path))
        movie_title_dict = load_dill_with_progress(
            path + '/movielens1m/data_processing/movie_title_dict_' + str(word_dim) + ".pkl"
            )
        print('load data successfully!')
    except:
        print('\nload movie_title_dict failed! creating new movie_title_dict...')
        movie_processor = processing_movie(movie=movie_new, path=path, word_dim=word_dim)
        movie_title_dict = movie_processor.get_movie_title_dict()
        # save
        print('\nsave movie_title_dict to {}/data_processing/movie_title_dict.pkl...'.format(path))
        with open(path + '/movielens1m/data_processing/movie_title_dict_' + str(word_dim) + '.pkl', 'wb') as f:
            dill.dump(movie_title_dict, f)

    # split train ,valid and test
    print('\nsplit train ,valid and test...')
    start_time = datetime.now()
    train_data, valid_data, test_data = split_train_test(behaviors_new=behaviors_new, train_ratio=0.6)
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    print('split done! cost {}s!'.format((datetime.now() - start_time).seconds))
    print('\ntrain_data shape:{}'.format(train_data.shape))
    print('valid_data shape:{}'.format(valid_data.shape))
    print('test_data shape:{}'.format(test_data.shape))

    # create dataset
    MyDataSet_train = MyDataSet(
        behaviors=train_data, movie=movie_new, movie_title_dict=movie_title_dict, history_len=history_len
        )
    MyDataSet_valid = MyDataSet(
        behaviors=valid_data, movie=movie_new, movie_title_dict=movie_title_dict, history_len=history_len
        )
    MyDataSet_test = MyDataSet(
        behaviors=test_data, movie=movie_new, movie_title_dict=movie_title_dict, history_len=history_len
        )

    weights = train_data['weight'].values
    DataLoader_train = DataLoader(
        MyDataSet_train,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(
            weights, num_samples=len(MyDataSet_train),
            replacement=True
            ),
        num_workers=num_workers,
        )

    DataLoader_valid = DataLoader(MyDataSet_valid, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    DataLoader_test = DataLoader(MyDataSet_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dic = {
        'train_loader': DataLoader_train,
        'valid_loader': DataLoader_valid,
        'test_loader': DataLoader_test,
        'items_new': movie_new,
        'behaviors_new': behaviors_new
        }

    return dic
