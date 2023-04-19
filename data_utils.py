import os
import pandas as pd
import numpy as np
import scipy.sparse as sp

from time import time
from collections import defaultdict


def create_adj_mat(mat, user_num, item_num, path, dataset, mode):
    t1 = time()
    adj_mat = sp.dok_matrix((user_num + item_num, user_num + item_num), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = mat
    adj_mat[:user_num, user_num:] = R
    adj_mat[user_num:, :user_num] = R.T
    adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape, time() - t1)

    t2 = time()

    def mean_adj_single(adj):
        # D^-1 * A
        rowsum = np.array(adj.sum(1))

        # d_inv = np.power(rowsum, -1).flatten()
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        norm_adj = norm_adj.dot(d_mat_inv)
        norm_adj = norm_adj.tocsr()
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        # return norm_adj.tocoo()
        return norm_adj.tocsr()

    norm_adj_mat = mean_adj_single(adj_mat)
    print('already normalize adjacency matrix in %.4fs' % (time() - t2))
    sp.save_npz(path + f'{dataset}_s_pre_adj_mat_{mode}.npz', norm_adj_mat)
    return norm_adj_mat


def load_pretrain(data_path, dataset):
    train_pv = os.path.join(data_path, '{}_pv_train.csv'.format(dataset))
    test_pv = os.path.join(data_path, '{}_pv_test.csv'.format(dataset))
    train_buy = os.path.join(data_path, '{}_buy_train.csv'.format(dataset))
    test_buy = os.path.join(data_path, '{}_buy_test.csv'.format(dataset))

    user_pos_dict_buy = defaultdict(list)
    user_pos_dict_pv = defaultdict(list)

    ################# load pv train data #################
    train_data = pd.read_csv(train_pv, sep='\t', header=0, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    train_aux = pd.read_csv(train_buy, sep='\t', header=0, names=['user', 'item'],
                            usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    train_data = pd.concat([train_data, train_aux], ignore_index=True)
    train_data.drop_duplicates(inplace=True, ignore_index=True)

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1
    num_interaction_pv = len(train_data)
    print(f"pv: user_num:{user_num}, item_num:{item_num}, interaction:{num_interaction_pv}")

    train_data_pv = train_data.values.tolist()

    train_data_dict_pv = defaultdict(list)
    train_mat_pv = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data_pv:
        uid = x[0]
        iid = x[1]
        train_mat_pv[uid, iid] = 1.0
        train_data_dict_pv[uid].append(iid)
        user_pos_dict_pv[uid].append(iid)

    ################# load buy train data #################
    train_data = pd.read_csv(train_buy, sep='\t', header=0, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    num_interaction_buy = len(train_data)
    print(f"buy: user_num:{user_num}, item_num:{item_num}, interaction:{num_interaction_buy}")

    train_data_buy = train_data.values.tolist()

    train_data_dict_buy = defaultdict(list)
    train_mat_buy = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data_buy:
        uid = x[0]
        iid = x[1]
        train_mat_buy[uid, iid] = 1.0
        train_data_dict_buy[uid].append(iid)
        user_pos_dict_buy[uid].append(iid)

    ################# load pv test data #################
    test_data = pd.read_csv(test_pv, sep='\t', header=0, names=['user', 'item'],
                            usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    print(f"number of pv test: {len(test_data)}")

    test_data_pv = test_data.values.tolist()

    test_data_dict_pv = defaultdict(list)
    for x in test_data_pv:
        uid = x[0]
        iid = x[1]
        test_data_dict_pv[uid].append(iid)

    ################# load buy test data #################
    test_data = pd.read_csv(test_buy, sep='\t', header=0, names=['user', 'item'],
                            usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    print(f"number of buy test: {len(test_data)}")

    test_data_buy = test_data.values.tolist()

    test_data_dict_buy = defaultdict(list)
    for x in test_data_buy:
        uid = x[0]
        iid = x[1]
        test_data_dict_buy[uid].append(iid)

    assert len(train_data_dict_pv) == len(train_data_dict_buy)

    return user_num, item_num, \
           train_mat_pv, train_mat_buy, \
           user_pos_dict_pv, user_pos_dict_buy,\
           train_data_dict_pv, train_data_dict_buy, \
           test_data_dict_pv, test_data_dict_buy


def load_all(data_path, dataset):
    train_pv = os.path.join(data_path, '{}_pv_train.csv'.format(dataset))
    test_pv = os.path.join(data_path, '{}_pv_test.csv'.format(dataset))
    train_buy = os.path.join(data_path, '{}_buy_train.csv'.format(dataset))
    test_buy = os.path.join(data_path, '{}_buy_test.csv'.format(dataset))

    user_pos_dict_buy = defaultdict(list)
    user_pos_dict_pv = defaultdict(list)

    # 加载点击训练集和测试集，并合并，同时确保购买训练集中的商品在用户的点击中出现
    train_data = pd.read_csv(train_pv, sep='\t', header=0, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    test_data = pd.read_csv(test_pv, sep='\t', header=0, names=['user', 'item'],
                            usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    train_aux = pd.read_csv(train_buy, sep='\t', header=0, names=['user', 'item'],
                            usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    train_data = pd.concat([train_data, test_data, train_aux], ignore_index=True)
    train_data.drop_duplicates(inplace=True, ignore_index=True)

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1
    num_interaction_pv = len(train_data)
    print(f"pv: user_num:{user_num}, item_num:{item_num}, interaction:{num_interaction_pv}")

    train_data_pv = train_data.values.tolist()

    train_data_dict_pv = defaultdict(list)
    train_mat_pv = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data_pv:
        uid = x[0]
        iid = x[1]
        train_mat_pv[uid, iid] = 1.0
        train_data_dict_pv[uid].append(iid)
        user_pos_dict_pv[uid].append(iid)

    ################# load buy training data #################
    train_data = pd.read_csv(train_buy, sep='\t', header=0, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    num_interaction_buy = len(train_data)
    print(f"buy: user_num:{user_num}, item_num:{item_num}, interaction:{num_interaction_buy}")

    train_data_buy = train_data.values.tolist()

    train_data_dict_buy = defaultdict(list)
    train_mat_buy = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data_buy:
        uid = x[0]
        iid = x[1]
        train_mat_buy[uid, iid] = 1.0
        train_data_dict_buy[uid].append(iid)
        user_pos_dict_buy[uid].append(iid)

    ################# load buy testing data #################
    test_data = pd.read_csv(test_buy, sep='\t', header=0, names=['user', 'item'],
                            usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    print(f"number of buy test: {len(test_data)}")

    test_data_buy = test_data.values.tolist()

    test_data_dict_buy = defaultdict(list)
    for x in test_data_buy:
        uid = x[0]
        iid = x[1]
        test_data_dict_buy[uid].append(iid)

    assert len(train_data_dict_pv) == len(train_data_dict_buy)

    return user_num, item_num, \
           train_mat_pv, train_mat_buy, \
           user_pos_dict_pv, user_pos_dict_buy,\
           train_data_dict_pv, train_data_dict_buy, \
           test_data_dict_buy
