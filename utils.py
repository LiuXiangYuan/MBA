import torch
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', early_stop_rounds=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= early_stop_rounds:
        print("Early stopping is trigger at step: {} log:{}".format(early_stop_rounds, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')
