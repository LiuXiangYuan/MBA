import torch
import copy
import torch.utils.data as data

from random import sample


def train_collate_fn(batch_dataset: list):
    return_tuple = [[], [], []]
    for example in batch_dataset:
        return_tuple[0].extend([[e] for e in example[0]])
        return_tuple[1].extend([[e] for e in example[1]])
        return_tuple[2].extend([[e] for e in example[2]])
    return_tuple[0] = torch.tensor(return_tuple[0], dtype=torch.long)
    return_tuple[1] = torch.tensor(return_tuple[1], dtype=torch.long)
    return_tuple[2] = torch.tensor(return_tuple[2], dtype=torch.long)
    return_tuple = tuple(return_tuple)
    return return_tuple


class MultiDataset(data.Dataset):
    def __init__(self,
                 user_num,
                 item_num,
                 train_data_dict_pv=None,
                 train_data_dict_buy=None,
                 train_mat_pv=None,
                 train_mat_buy=None,
                 test_data_dict_pv=None,
                 test_data_dict_buy=None,
                 test_mat_pv=None,
                 test_mat_buy=None,
                 user_pos_dict_pv=None,
                 user_pos_dict_buy=None,
                 test_pv_or_buy="buy",
                 train_mode="pv",
                 is_pretrain=False):
        super(MultiDataset, self).__init__()

        self.test_pv_or_buy = test_pv_or_buy
        self.train_mode = train_mode
        self.is_pretrain = is_pretrain

        self.user_num = user_num
        self.item_num = item_num

        self.train_data_dict_pv = train_data_dict_pv
        self.train_mat_pv = train_mat_pv
        self.train_data_dict_buy = train_data_dict_buy
        self.train_mat_buy = train_mat_buy

        self.test_data_dict_pv = test_data_dict_pv
        self.test_mat_pv = test_mat_pv
        self.test_data_dict_buy = test_data_dict_buy
        self.test_mat_buy = test_mat_buy

        self.rest_pv_dict = None

        self.user_pos_dict_pv = user_pos_dict_pv
        self.user_pos_dict_buy = user_pos_dict_buy

        self.features = None
        self.features_pv = None
        self.features_buy = None

        self.build_features()

    def build_features(self):
        # features 的格式统一为[uid, pv_iid, buy_iid]
        self.features = []  # 取都为正例的example
        self.features_pv = []  # 取pv所有正例，并随机抽取buy的正例
        self.features_buy = []  # 取buy所有正例，并随机抽取pv的正例

        train_data_dict_pv = copy.deepcopy(self.train_data_dict_pv)
        train_data_dict_buy = copy.deepcopy(self.train_data_dict_buy)

        if self.is_pretrain:
            for uid, buy_list in train_data_dict_buy.items():
                pv_list = train_data_dict_pv[uid]
                for buy_iid in buy_list:
                    pv_iid = sample(pv_list, 1)[0]
                    feature = [uid, pv_iid, buy_iid]
                    self.features_buy.append(feature)

            for uid, pv_list in train_data_dict_pv.items():
                buy_list = train_data_dict_buy[uid]
                for pv_iid in pv_list:
                    buy_iid = sample(buy_list, 1)[0]
                    feature = [uid, pv_iid, buy_iid]
                    self.features_pv.append(feature)
        else:
            for uid, buy_list in train_data_dict_buy.items():
                pv_list = train_data_dict_pv[uid]
                for buy_iid in buy_list:
                    feature = [uid, buy_iid, buy_iid]
                    if buy_iid in pv_list:
                        pv_list.remove(buy_iid)
                    self.features.append(feature)

            self.rest_pv_dict = train_data_dict_pv
            for uid, pv_list in train_data_dict_pv.items():
                if len(pv_list) == 0:
                    continue
                for pv_iid in pv_list:
                    feature = [uid, pv_iid, pv_iid]
                    self.features_pv.append(feature)

    def __len__(self):
        if self.train_mode == "both":
            return len(self.features)
        elif self.train_mode == "buy":
            return len(self.features_buy)
        else:
            return len(self.features_pv)

    def __getitem__(self, idx):
        # features 的格式统一为[uid, pv_iid, buy_iid]
        if self.train_mode == "both":
            feature = self.features[idx]
        elif self.train_mode == "pv":
            feature = self.features_pv[idx]
        else:
            feature = self.features_buy[idx]
        uid, pv, buy = feature
        return [uid], [pv], [buy]
