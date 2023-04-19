import argparse
import os

import evaluate
import numpy as np
import scipy.sparse as sp
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from utils import set_seed
from datasets import MultiDataset
from datasets import train_collate_fn
from data_utils import create_adj_mat, load_all, load_pretrain

from model import MF
from LightGCN import LightGCN

from train import TrainModel


def main(param):
    seed = param['seed']
    set_seed(seed)
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # cudnn

    def worker_init_fn(worker_id):
        np.random.seed(param['seed'] + worker_id)

    if not os.path.exists(param['folder']):
        os.mkdir(param['folder'])

    ###################### PREPARE DATASET ##########################

    data_path = f"{param['datadir']}/{param['dataset']}/"
    # load data
    if param['train_method'] == "pre":
        user_num, item_num, \
        train_mat_pv, train_mat_buy, \
        user_pos_dict_pv, user_pos_dict_buy, \
        train_data_dict_pv, train_data_dict_buy, \
        test_data_dict_pv, test_data_dict_buy = load_pretrain(data_path, param['dataset'])
    else:
        user_num, item_num, \
        train_mat_pv, train_mat_buy, \
        user_pos_dict_pv, user_pos_dict_buy, \
        train_data_dict_pv, train_data_dict_buy, \
        test_data_dict_buy = load_all(data_path, param['dataset'])

    train_dataset = MultiDataset(user_num=user_num, item_num=item_num,
                                 train_mat_pv=train_mat_pv, train_mat_buy=train_mat_buy,
                                 user_pos_dict_pv=user_pos_dict_pv, user_pos_dict_buy=user_pos_dict_buy,
                                 train_data_dict_pv=train_data_dict_pv, train_data_dict_buy=train_data_dict_buy,
                                 is_pretrain=(param['train_method'] == "pre"))

    train_loader = data.DataLoader(train_dataset, batch_size=param.get('batch_size', 2048),
                                   shuffle=True, num_workers=0, pin_memory=True,
                                   worker_init_fn=worker_init_fn, collate_fn=train_collate_fn)

    ########################### CREATE MODEL #################################
    pretrain_model = param['pretrain_model']
    model_name = param['model']

    try:
        norm_adj_buy = sp.load_npz(data_path + param['dataset'] + '_s_pre_adj_mat_buy.npz')
        norm_adj_pv = sp.load_npz(data_path + param['dataset'] + '_s_pre_adj_mat_pv.npz')
        print("successfully loaded...")
    except:
        norm_adj_buy = create_adj_mat(train_dataset.train_mat_buy,
                                      train_dataset.user_num, train_dataset.item_num,
                                      data_path, dataset=param['dataset'], mode="buy")
        norm_adj_pv = create_adj_mat(train_dataset.train_mat_pv,
                                     train_dataset.user_num, train_dataset.item_num,
                                     data_path, dataset=param['dataset'], mode="pv")

    param['norm_adj_buy'] = norm_adj_buy
    param['norm_adj_pv'] = norm_adj_pv

    if pretrain_model == 'lgn':
        model_buy = LightGCN(train_dataset.user_num, train_dataset.item_num,
                             norm_adj_buy, latent_dim=param['emb_dim'], n_layers=param['num_layers'],
                             device=param['device'], dropout=param['dropout'])
        model_pv = LightGCN(train_dataset.user_num, train_dataset.item_num,
                            norm_adj_pv, latent_dim=param['emb_dim'], n_layers=param['num_layers'],
                            device=param['device'], dropout=param['dropout'])
    elif pretrain_model == 'MF':
        model_buy = MF(train_dataset.user_num, train_dataset.item_num,
                       param.get("factor_num", 32))
        model_pv = MF(train_dataset.user_num, train_dataset.item_num,
                       param.get("factor_num", 32))
    else:
        assert False, "模型未实现"

    train_model = TrainModel(param=param,
                             user_num=train_dataset.user_num, item_num=train_dataset.item_num,
                             train_mat_buy=train_dataset.train_mat_buy, train_mat_pv=train_dataset.train_mat_pv)

    data_name = param['dataset']
    lambda0 = str(param['lambda0'])
    idx = param['idx']
    pv_load_model_path = os.path.join(param['folder'],
                                      f"pretrain_pv_{pretrain_model}_{data_name}_{seed}_{lambda0}_{idx}.pt")
    buy_load_model_path = os.path.join(param['folder'],
                                       f"pretrain_buy_{pretrain_model}_{data_name}_{seed}_{lambda0}_{idx}.pt")

    if param['train_method'] == "pre":
        if param['test_only']:
            model_buy.load_state_dict(torch.load(buy_load_model_path, map_location=torch.device('cpu')))
            recall, NDCG = evaluate.test_all_users(model_buy, 4096, test_data_dict_buy, user_pos_dict_buy,
                                                   train_model.top_k, device=train_model.device)
            final_perf = "TEST\t recall=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.4f' % r for r in recall]),
                          '\t'.join(['%.4f' % r for r in NDCG]))
            print(final_perf)

        else:
            train_model.pretrain(model=model_pv, train_loader=train_loader,
                                  train_mode="pv",
                                  test_data_pos=test_data_dict_pv, user_pos=user_pos_dict_pv,
                                  model_name=pretrain_model, early_stop_rounds=param['pretrain_early_stop_rounds'],
                                  model_save_path=pv_load_model_path)
            set_seed(seed)
            train_model.pretrain(model=model_buy, train_loader=train_loader,
                                  train_mode="buy",
                                  test_data_pos=test_data_dict_buy, user_pos=user_pos_dict_buy,
                                  model_name=pretrain_model, early_stop_rounds=param['pretrain_early_stop_rounds'],
                                  model_save_path=buy_load_model_path)
    elif param['train_method'] == "mba":
        is_exist_pretrain = os.path.exists(pv_load_model_path)
        assert is_exist_pretrain, "点击预训练模型路径不存在"

        is_exist_pretrain = os.path.exists(buy_load_model_path)
        assert is_exist_pretrain, "购买预训练模型路径不存在"

        if param['test_only']:
            model_buy.load_state_dict(torch.load(buy_load_model_path, map_location=torch.device('cpu')))
            if pretrain_model == 'MF':
                target_buy = MF(train_model.user_num, train_model.item_num, param.get("factor_num", 32))
            else:
                target_buy = LightGCN(train_dataset.user_num, train_dataset.item_num,
                                      norm_adj_buy, latent_dim=param['emb_dim'], n_layers=param['num_layers'],
                                      device=param['device'], dropout=param['dropout'])
            target_model_path = os.path.join(param['folder'],
                                             f"{param['dataset']}_target_{model_name}_{seed}_{str(param['lambda1'])}.pt")
            target_buy.load_state_dict(torch.load(target_model_path, map_location=torch.device('cpu')))
            recall, NDCG = evaluate.test_all_users_with_two_model(target_buy, model_buy, 4096,
                                                                  test_data_dict_buy, user_pos_dict_buy, train_model.top_k,
                                                                  device=train_model.device,
                                                                  beta1=train_model.beta1, beta2=train_model.beta2)
            final_perf = "TEST\t recall=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.4f' % r for r in recall]),
                          '\t'.join(['%.4f' % r for r in NDCG]))
            print(final_perf)
        else:
            model_pv.load_state_dict(torch.load(pv_load_model_path, map_location=torch.device('cpu')))
            model_buy.load_state_dict(torch.load(buy_load_model_path, map_location=torch.device('cpu')))

            train_model.train(pv_pre_model=model_pv, buy_pre_model=model_buy, train_loader=train_loader,
                              test_data_pos=test_data_dict_buy, user_pos=user_pos_dict_buy,
                              data_source='buy', model_name=model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # flexible parameters
    parser.add_argument('--datadir', type=str, default=r'./data')
    parser.add_argument('--folder', type=str, default='./output')
    parser.add_argument('--model', type=str, default='MF')
    parser.add_argument('--h_model', type=str, default='MF')
    parser.add_argument('--dataset', type=str, default='beibei',
                        help='dataset used for training, options: beibei, taobao')
    parser.add_argument("--epochs", type=int, default=400, help="training epoches")
    parser.add_argument("--top_k", type=list, default=[10, 20], help="compute metrics@top_k")
    parser.add_argument("--C_1", default=1000, type=int, help='the large number used in DP')
    parser.add_argument("--C_2", default=1000, type=int, help='the large number used in DN')
    parser.add_argument("--alpha", type=float, default=1.0, help='weight between two KL divergence')
    parser.add_argument("--lambda0", type=float, default=1e-4, help='regularization parameter')
    parser.add_argument("--lambda1", type=float, default=1e-6, help='regularization parameter')
    parser.add_argument("--save_model", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2020)

    # following parameters are fixed during my implementation
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=2048, help="batch size for training")
    parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
    parser.add_argument("--num_layers", type=int, default=3, help="number of layers in MLP model")
    parser.add_argument("--NSR", type=int, default=1, help="sample negative items for training")
    parser.add_argument("--device", default='cuda', help='cuda or cpu')
    parser.add_argument("--emb_dim", type=int, default=32, help='embedding dimension of the Gamma model')
    parser.add_argument("--early_stop_rounds", type=int, default=30,
                        help='early stop after how many iteration rounds non decreasing')

    # 新加入的参数
    parser.add_argument("--train_method", type=str, default="mba", help="mba: MBA, pre: pretrain")
    parser.add_argument("--pretrain_early_stop_rounds", type=int, default=20)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--pretrain_model", type=str, default="MF")
    parser.add_argument("--denoise_type", type=str, default="DP")
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument('--test_only', action='store_true')

    args = parser.parse_args()

    main({
        'datadir': args.datadir,
        'folder': args.folder,
        'model': args.model,
        'h_model': args.h_model,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'top_k': args.top_k,
        'C_2': args.C_2,
        'C_1': args.C_1,
        'alpha': args.alpha,
        'lambda0': args.lambda0,
        'lambda1': args.lambda0,
        'save_model': args.save_model,
        "seed": args.seed,
        'dropout': args.dropout,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'factor_num': args.factor_num,
        'num_layers': args.num_layers,
        'NSR': args.NSR,
        'device': args.device,
        'emb_dim': args.emb_dim,
        'early_stop_rounds': args.early_stop_rounds,
        # 新加入的参数
        'train_method': args.train_method,
        'pretrain_early_stop_rounds': args.pretrain_early_stop_rounds,
        'idx': args.idx,
        'pretrain_model': args.pretrain_model,
        'denoise_type': args.denoise_type,
        'beta1': args.beta,
        'beta2': 1.0 - args.beta,
        'test_only': args.test_only
    })
