import torch
import torch.nn.functional as F


def model_forward(model, users, items, detach=False):
    if detach:
        with torch.no_grad():
            return model(users, items)
    else:
        return model(users, items)


def log(x):
    return torch.log(x + 1e-5)


def KL_loss(prediction_A, prediction_B, beta=1.0):
    prediction_A = torch.sigmoid(prediction_A)
    prediction_B = torch.sigmoid(prediction_B)

    loss = KL(prediction_A, prediction_B) * beta + \
           KL(prediction_B, prediction_A) * (1.0 - beta)

    return loss


def KL(p1, p2):
    return p1 * log(p1) - p1 * log(p2) + \
           (1 - p1) * log(1 - p1) - (1 - p1) * log(1 - p2)


def bpr_loss(pos_scores, neg_scores):
    loss = torch.mean(F.softplus(neg_scores - pos_scores))
    return loss


def denoise_positive_loss(prediction_h1, prediction_h2, prediction_target, C=None, denoise_type=None):
    prediction_h1 = torch.sigmoid(prediction_h1)
    if denoise_type != "DN":
        prediction_h2 = torch.sigmoid(prediction_h2)
    prediction_target = torch.sigmoid(prediction_target)

    if denoise_type == "DN":
        return log(prediction_h1) * prediction_target - \
               C * (1 - prediction_target)
    elif denoise_type == "DP":
        return log(prediction_h2) * (1 - prediction_target)
    else:
        return log(prediction_h1) * prediction_target + \
               log(prediction_h2) * (1 - prediction_target)


def denoise_negative_loss(prediction_h1, prediction_h2, prediction_target, C=None, denoise_type=None):
    prediction_h1 = torch.sigmoid(prediction_h1)
    if denoise_type != "DN":
        prediction_h2 = torch.sigmoid(prediction_h2)
    prediction_target = torch.sigmoid(prediction_target)

    if denoise_type == "DN":
        return log(1 - prediction_h1) * prediction_target
    elif denoise_type == "DP":
        return log(1 - prediction_h2) * (1 - prediction_target) - \
               C * prediction_target
    else:
        return log(1 - prediction_h1) * prediction_target + \
               log(1 - prediction_h2) * (1 - prediction_target)
