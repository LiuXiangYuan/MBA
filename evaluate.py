import numpy as np
import torch

from utils import getLabel
from metrics import RecallPrecision_ATk, NDCGatK_r


def getUserPosItems(allPos, users):
    if isinstance(users, int):
        users = [users]

    posItems = []
    for user in users:
        posItems.append(allPos[user])

    return posItems


def test_one_batch(Ks, X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in Ks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'precision': np.array(pre),
            'recall': np.array(recall),
            'ndcg': np.array(ndcg)}


def test_all_users(model, batch_size, test_data_pos, user_pos, top_k, device='cuda'):

    model.eval()
    model.to(device)
    max_K = max(top_k)

    result = {'precision': np.zeros(len(top_k)),
              'recall': np.zeros(len(top_k)),
              'ndcg': np.zeros(len(top_k))}

    test_users = np.array(list(test_data_pos.keys()))
    ground_true_items = list(test_data_pos.values())

    n_test = len(test_users)

    try:
        assert batch_size <= n_test / 10
    except AssertionError:
        print(f"test_batch_size is too big for this dataset, try a small one {n_test // 10}")
        batch_size = n_test // 10

    n_batchs = n_test // batch_size + 1

    rating_list = []
    groundTrue_list = []

    with torch.no_grad():
        for u_batch_id in range(n_batchs):
            start = u_batch_id * batch_size
            end = (u_batch_id + 1) * batch_size

            user_batch = test_users[start: end]
            if len(user_batch) == 0:
                continue

            allPos = getUserPosItems(user_pos, user_batch)
            groundTrue = ground_true_items[start: end]

            batch_users_gpu = torch.Tensor(user_batch).long()
            batch_users_gpu = batch_users_gpu.to(device)

            rating = model.getUsersRating(batch_users_gpu)

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                if len(items) == 0:
                    continue
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)

            _, rating_K = torch.topk(rating, k=max_K)

            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(top_k, x))

        for re in pre_results:
            result['recall'] += re['recall']
            result['ndcg'] += re['ndcg']
        result['recall'] /= float(n_test)
        result['ndcg'] /= float(n_test)

        recall = result['recall']
        NDCG = result['ndcg']

        return recall, NDCG


def test_all_users_with_two_model(model, h_model, batch_size, test_data_pos, user_pos, top_k,
                                  device='cuda', beta1=1.0, beta2=1.0):

    model.eval()
    model.to(device)
    h_model.eval()
    h_model.to(device)
    max_K = max(top_k)

    result = {'precision': np.zeros(len(top_k)),
              'recall': np.zeros(len(top_k)),
              'ndcg': np.zeros(len(top_k))}

    test_users = np.array(list(test_data_pos.keys()))
    ground_true_items = list(test_data_pos.values())

    n_test = len(test_users)

    try:
        assert batch_size <= n_test / 10
    except AssertionError:
        print(f"test_batch_size is too big for this dataset, try a small one {n_test // 10}")
        batch_size = n_test // 10

    n_batchs = n_test // batch_size + 1

    rating_list = []
    groundTrue_list = []

    with torch.no_grad():
        for u_batch_id in range(n_batchs):
            start = u_batch_id * batch_size
            end = (u_batch_id + 1) * batch_size

            user_batch = test_users[start: end]
            if len(user_batch) == 0:
                continue

            allPos = getUserPosItems(user_pos, user_batch)
            groundTrue = ground_true_items[start: end]

            batch_users_gpu = torch.Tensor(user_batch).long()
            batch_users_gpu = batch_users_gpu.to(device)

            rating = model.getUsersRating(batch_users_gpu)
            rating_h = h_model.getUsersRating(batch_users_gpu)
            rating = beta1 * rating + beta2 * rating_h

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                if len(items) == 0:
                    continue
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)

            _, rating_K = torch.topk(rating, k=max_K)

            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(top_k, x))

        for re in pre_results:
            result['recall'] += re['recall']
            result['ndcg'] += re['ndcg']
        result['recall'] /= float(n_test)
        result['ndcg'] /= float(n_test)

        recall = result['recall']
        NDCG = result['ndcg']

        return recall, NDCG
