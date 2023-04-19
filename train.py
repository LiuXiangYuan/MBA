import os
import random

import torch
import torch.optim as optim
import numpy as np
import evaluate

from LightGCN import LightGCN
from model import MF
from utils import early_stopping

from loss_func import (
    model_forward,
    bpr_loss,
    KL_loss,
    denoise_positive_loss,
    denoise_negative_loss
)


class TrainModel:
    def __init__(self, param, user_num, item_num, train_mat_buy, train_mat_pv):
        self.param = param
        self.model_name = self.param['model']
        self.user_num = user_num
        self.item_num = item_num
        self.train_mat_buy = train_mat_buy
        self.train_mat_pv = train_mat_pv
        self.device = param.get("device", 'cuda')
        self.seed = param['seed']
        self.top_k = param['top_k']
        self.C_1 = param.get('C_1', 1000)
        self.C_2 = param.get('C_2', 1000)
        self.NSR = self.param.get("NSR", 1)
        self.alpha = self.param.get("alpha", 1)
        self.early_stop_rounds = param.get("early_stop_rounds", 5)
        self.emb_dim = param.get("emb_dim", 32)
        self.h_model = param.get("h_model", 'MF')
        self.lr = param.get('lr', 0.001)
        self.denoise_type = param.get("denoise_type", 'both')
        self.beta1 = param.get('beta1', 1.0)
        self.beta2 = param.get('beta2', 1.0)

    def print_pretrain_config(self, train_mode: str, model_name: str, early_stop_rounds: int):
        print(f"###### pretrain {train_mode} config #######")
        print(f"seed: {self.seed}")
        print(f"model: {model_name}, dataset: {self.param['dataset']}")
        print(f"epochs: {self.param['epochs']}, early_stop_rounds: {early_stop_rounds}")
        print(f"regularization lambda: {self.param['lambda0']}")
        print(f"negative sampling rate: {self.NSR}")

    def print_train_config(self, model_name, data_source):
        print(f"###### one way train config #######")
        print(f"seed: {self.seed}")
        print(f"model: {model_name}, dataset: {self.param['dataset']}")
        print("h_model:", self.h_model)
        print("alpha:", self.alpha)
        print(f"C_1 is: {self.C_1}, C_2 is: {self.C_2}")
        print(f"epochs: {self.param['epochs']}, early_stop_rounds: {self.early_stop_rounds}")
        print(f"regularization lambda: {self.param['lambda0']}")
        print(f"negative sampling rate: {self.NSR}")
        print(f"train model base on: {data_source}")

    def get_model(self, data_source='pv', get_h=False):
        if get_h:
            model_name = self.h_model
        else:
            model_name = self.model_name

        if model_name == 'lgn':
            if data_source == 'pv':
                norm_adj = self.param['norm_adj_pv']
            else:
                norm_adj = self.param['norm_adj_buy']

            model = LightGCN(self.user_num, self.item_num,
                             norm_adj, latent_dim=self.emb_dim, n_layers=self.param['num_layers'],
                             device=self.param['device'], dropout=self.param['dropout']).to(self.param['device'])
        else:
            model = MF(user_num=self.user_num, item_num=self.item_num, K0=self.emb_dim)

        return model

    def forward(self, model, user, pos_item, NSR=1, source='buy', neg_item=None, detach=False):
        """
        :param model: model
        :param user: user
        :param pos_item: positive items tensor: (bsz, )
        :param NSR: negative sample rate
        :param source: data source
        :param neg_item: neg_items
        :param detach: is_detach
        :return: positive scores and negative scores
        """
        if neg_item is None:
            neg_item = self.sample_neg_items(user, source)

        pos_prediction_logits = model_forward(model, user, pos_item, detach=detach)
        neg_prediction_logits = model_forward(model, user, neg_item, detach=detach)

        if NSR > 1:
            for i in range(NSR - 1):
                new_neg_item = self.sample_neg_items(user, source)
                neg_prediction_logits = torch.cat([neg_prediction_logits,
                                                   model_forward(model, user, new_neg_item, detach=detach)],
                                                  dim=0)
                neg_item = torch.cat([neg_item, new_neg_item], dim=0)

        if detach:
            pos_prediction_logits.detach()
            neg_prediction_logits.detach()

        return pos_prediction_logits, neg_prediction_logits, neg_item

    def sample_neg_items(self, user, source):
        neg_item = []
        for single_user in user:
            j = self.random_choice()
            if source == "both":
                while (single_user, j) in self.train_mat_buy or (single_user, j) in self.train_mat_pv:
                    j = self.random_choice()
            else:
                if source == "buy":
                    train_mat = self.train_mat_buy
                else:
                    train_mat = self.train_mat_pv

                while (single_user, j) in train_mat:
                    j = self.random_choice()
            neg_item.append(j)
        neg_item = torch.tensor(neg_item).long().to(self.device)
        return neg_item

    def sample_rest_pv_items(self, user, rest_pv_dict):
        rest_item = []
        for single_user in user:
            j = random.choice(rest_pv_dict[single_user.item()])
            rest_item.append(j)
        rest_item = torch.tensor(rest_item).long().to(self.device)
        return rest_item

    def random_choice(self):
        return np.random.randint(self.item_num)

    def pretrain(self, model, train_loader,train_mode,
                 test_data_pos, user_pos,
                 model_name, early_stop_rounds, model_save_path=None):

        self.print_pretrain_config(train_mode, model_name, early_stop_rounds)

        count, last_loss = 0, 1e9
        cur_best_pre_0 = 0.
        stopping_step = 0
        epochs = self.param['epochs']
        rec_loger, ndcg_loger = [], []

        pretrain_model_optim = optim.Adam(model.parameters(), lr=self.lr)
        model.to(self.device)
        for epoch in range(epochs):
            model.train()

            train_loader.dataset.train_mode = train_mode
            for uid, pos_pv, pos_buy in train_loader:
                if train_mode == "pv":
                    pos = pos_pv
                else:
                    pos = pos_buy
                pretrain_model_optim.zero_grad()
                uid, pos = uid.squeeze(1), pos.squeeze(1)
                uid, pos = uid.to(self.device), pos.to(self.device)
                pos_prediction_logits, neg_prediction_logits, neg_item = \
                    self.forward(model, uid, pos, self.NSR, source=train_mode)

                loss = bpr_loss(pos_prediction_logits, neg_prediction_logits)

                reg_loss = 0
                for param in model.parameters():
                    reg_loss += param.norm(2).pow(2)

                reg_loss = 1 / 2 * reg_loss / float(self.user_num)

                loss += reg_loss * self.param['lambda0']
                loss.backward()

                pretrain_model_optim.step()

                if count % 200 == 0 and count != 0:
                    print(f"pretrain {train_mode} model epoch: {epoch}, iter: {count}, loss:{loss}")
                count += 1

            print("################### PRETRAIN TEST ######################")
            recall, NDCG = evaluate.test_all_users(model, 4096, test_data_pos, user_pos, self.top_k, device=self.device)
            final_perf = "Iter=[%d]\t recall=[%s], ndcg=[%s]" % \
                         (epoch,
                          '\t'.join(['%.4f' % r for r in recall]),
                          '\t'.join(['%.4f' % r for r in NDCG]))
            print(final_perf)

            rec_loger.append(recall)
            ndcg_loger.append(NDCG)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(recall[-1],
                                                                        cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        early_stop_rounds=early_stop_rounds)
            if should_stop:
                print("pretrain early stop.")
                break

            if cur_best_pre_0 == recall[-1]:
                if model_save_path is None:
                    model_save_path = os.path.join(self.param['folder'], f"pretrain_{train_mode}_{model_name}_{self.param['dataset']}_{self.seed}_{str(self.param['lambda0'])}.pt")
                print("saving pretrain model...")
                print(f"save to {model_save_path}")
                torch.save(model.state_dict(), model_save_path)

        recs = np.array(rec_loger)
        ndcgs = np.array(ndcg_loger)

        best_rec_0 = max(recs[:, -1])
        best_idx = list(recs[:, -1]).index(best_rec_0)

        final_perf = "Best Iter=[%d]\t recall=[%s], ndcg=[%s]" % \
                     (best_idx,
                      '\t'.join(['%.5f' % r for r in recs[best_idx]]),
                      '\t'.join(['%.5f' % r for r in ndcgs[best_idx]]))
        print(final_perf)

    def train(self, pv_pre_model, buy_pre_model,
                    train_loader, test_data_pos,
                    user_pos, data_source, model_name):

        self.print_train_config(model_name, data_source)

        pv_h1_model = self.get_model(data_source="pv", get_h=True)
        pv_h2_model = self.get_model(data_source="pv", get_h=True)
        buy_h1_model = self.get_model(data_source="buy", get_h=True)
        buy_h2_model = self.get_model(data_source="buy", get_h=True)
        target_model = self.get_model(data_source=data_source)

        pv_h1_model_optim = optim.Adam(pv_h1_model.parameters(), lr=self.lr)
        pv_h2_model_optim = optim.Adam(pv_h2_model.parameters(), lr=self.lr)
        buy_h1_model_optim = optim.Adam(buy_h1_model.parameters(), lr=self.lr)
        buy_h2_model_optim = optim.Adam(buy_h2_model.parameters(), lr=self.lr)
        target_model_optim = optim.Adam(target_model.parameters(), lr=self.lr)

        epochs = self.param['epochs']

        pv_h1_model.to(self.device)
        pv_h2_model.to(self.device)
        buy_h1_model.to(self.device)
        buy_h2_model.to(self.device)
        target_model.to(self.device)
        pv_pre_model.to(self.device)
        buy_pre_model.to(self.device)

        def _on_iteration_start():
            pv_h1_model_optim.zero_grad()
            pv_h2_model_optim.zero_grad()
            buy_h1_model_optim.zero_grad()
            buy_h2_model_optim.zero_grad()
            target_model_optim.zero_grad()

        def _on_iteration_end():
            if self.denoise_type != 'DP':
                pv_h1_model_optim.step()
                buy_h1_model_optim.step()
            if self.denoise_type != 'DN':
                pv_h2_model_optim.step()
                buy_h2_model_optim.step()
            target_model_optim.step()

            if self.denoise_type == 'DP':
                self.denoise_type = 'DN'
            elif self.denoise_type == 'DN':
                self.denoise_type = 'DP'

        count, last_loss = 0, 1e9
        cur_best_pre_0 = 0.
        stopping_step = 0
        both_rec_loger, both_ndcg_loger = [], []

        for epoch in range(epochs):
            pv_h1_model.train()
            pv_h2_model.train()
            buy_h1_model.train()
            buy_h2_model.train()
            target_model.train()
            pv_pre_model.eval()
            buy_pre_model.eval()

            # 点击并且购买的sample
            train_loader.dataset.train_mode = "both"
            for uid, pos_item, _ in train_loader:

                _on_iteration_start()
                uid, pos_item = uid.squeeze(1), pos_item.squeeze(1)
                uid, pos_item = uid.to(self.device), pos_item.to(self.device)

                pos_prediction_buy_h1, neg_prediction_buy_h1, neg_items_both = \
                    self.forward(buy_h1_model, uid, pos_item, self.NSR, source="both")
                pos_prediction_buy_h2, neg_prediction_buy_h2, _ = \
                    self.forward(buy_h2_model, uid, pos_item, self.NSR, source="both", neg_item=neg_items_both)
                pos_prediction_target, neg_prediction_target, _ = \
                    self.forward(target_model, uid, pos_item, self.NSR, source="both", neg_item=neg_items_both)
                pos_prediction_pv_pretrain, neg_prediction_pv_pretrain, _ = \
                    self.forward(pv_pre_model, uid, pos_item, self.NSR, source="both", detach=True, neg_item=neg_items_both)

                denoise_pos_loss_buy = denoise_positive_loss(pos_prediction_buy_h1, pos_prediction_buy_h2,
                                                             pos_prediction_target, C=self.C_1, denoise_type=self.denoise_type)
                denoise_neg_loss_buy = denoise_negative_loss(neg_prediction_buy_h1, neg_prediction_buy_h2,
                                                             neg_prediction_target, C=self.C_2, denoise_type=self.denoise_type)

                kl_pos_loss_pv = self.alpha * KL_loss(pos_prediction_pv_pretrain, pos_prediction_target)
                kl_neg_loss_pv = self.alpha * KL_loss(neg_prediction_pv_pretrain, neg_prediction_target)

                pos_loss_buy = torch.mean(kl_pos_loss_pv - denoise_pos_loss_buy)
                neg_loss_buy = torch.mean(kl_neg_loss_pv - denoise_neg_loss_buy)

                pos_prediction_pv_h1, neg_prediction_pv_h1, _ = \
                    self.forward(pv_h1_model, uid, pos_item, self.NSR, source="both", neg_item=neg_items_both)
                pos_prediction_pv_h2, neg_prediction_pv_h2, _ = \
                    self.forward(pv_h2_model, uid, pos_item, self.NSR, source="both", neg_item=neg_items_both)
                pos_prediction_buy_pretrain, neg_prediction_buy_pretrain, _ = \
                    self.forward(buy_pre_model, uid, pos_item, self.NSR, source="both", detach=True, neg_item=neg_items_both)

                denoise_pos_loss_pv = denoise_positive_loss(pos_prediction_pv_h1, pos_prediction_pv_h2,
                                                            pos_prediction_target, C=self.C_1, denoise_type=self.denoise_type)
                denoise_neg_loss_pv = denoise_negative_loss(neg_prediction_pv_h1, neg_prediction_pv_h2,
                                                            neg_prediction_target, C=self.C_2, denoise_type=self.denoise_type)

                kl_pos_loss_buy = self.alpha * KL_loss(pos_prediction_buy_pretrain, pos_prediction_target)
                kl_neg_loss_buy = self.alpha * KL_loss(neg_prediction_buy_pretrain, neg_prediction_target)

                pos_loss_pv = torch.mean(kl_pos_loss_buy - denoise_pos_loss_pv)
                neg_loss_pv = torch.mean(kl_neg_loss_buy - denoise_neg_loss_pv)

                loss = pos_loss_buy + neg_loss_buy + pos_loss_pv + neg_loss_pv

                reg_loss = 0
                for param in target_model.parameters():
                    reg_loss += param.norm(2).pow(2)

                reg_loss = 1 / 2 * reg_loss / float(self.user_num)

                loss += reg_loss * self.param['lambda1']
                loss.backward()
                _on_iteration_end()

                if count % 200 == 0 and count != 0:
                    print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))
                count += 1

            # 剩下的点击但是没有购买的sample
            train_loader.dataset.train_mode = "pv"
            for uid, pos_pv, _ in train_loader:

                _on_iteration_start()
                uid, pos_pv = uid.squeeze(1), pos_pv.squeeze(1)
                uid, pos_pv = uid.to(self.device), pos_pv.to(self.device)

                pos_prediction_pv_h1, neg_prediction_pv_h1, neg_items_both = \
                    self.forward(pv_h1_model, uid, pos_pv, self.NSR, source="both")
                pos_prediction_pv_h2, neg_prediction_pv_h2, _ = \
                    self.forward(pv_h2_model, uid, pos_pv, self.NSR, source="both", neg_item=neg_items_both)
                pos_prediction_target, neg_prediction_target, _ = \
                    self.forward(target_model, uid, pos_pv, self.NSR, source="both", neg_item=neg_items_both)
                pos_prediction_pv_pretrain, neg_prediction_pv_pretrain, _ = \
                    self.forward(pv_pre_model, uid, pos_pv, self.NSR, source="both", detach=True, neg_item=neg_items_both)

                neg_prediction_buy_pretrain_rest, neg_prediction_buy_pretrain_both, _ = \
                    self.forward(buy_pre_model, uid, pos_pv, self.NSR, source="both", detach=True, neg_item=neg_items_both)
                neg_prediction_buy_h1_rest, neg_prediction_buy_h1_both, _ = \
                    self.forward(buy_h1_model, uid, pos_pv, self.NSR, source="both", neg_item=neg_items_both)
                neg_prediction_buy_h2_rest, neg_prediction_buy_h2_both, _ = \
                    self.forward(buy_h2_model, uid, pos_pv, self.NSR, source="both", neg_item=neg_items_both)

                denoise_pos_loss_pv = denoise_positive_loss(pos_prediction_pv_h1, pos_prediction_pv_h2,
                                                            pos_prediction_target, C=self.C_1, denoise_type=self.denoise_type)
                denoise_neg_loss_pv = denoise_negative_loss(neg_prediction_pv_h1, neg_prediction_pv_h2,
                                                            neg_prediction_target, C=self.C_2, denoise_type=self.denoise_type)

                denoise_neg_loss_buy_rest = denoise_negative_loss(neg_prediction_buy_h1_rest, neg_prediction_buy_h2_rest,
                                                                  pos_prediction_target, C=self.C_2, denoise_type=self.denoise_type)
                denoise_neg_loss_buy_both = denoise_negative_loss(neg_prediction_buy_h1_both, neg_prediction_buy_h2_both,
                                                                  neg_prediction_target, C=self.C_2, denoise_type=self.denoise_type)

                kl_neg_loss_rest = self.alpha * KL_loss(neg_prediction_buy_pretrain_rest, pos_prediction_target)
                kl_neg_loss_both = self.alpha * KL_loss(neg_prediction_buy_pretrain_both, neg_prediction_target)

                kl_pos_loss_pv = self.alpha * KL_loss(pos_prediction_pv_pretrain, pos_prediction_target)
                kl_neg_loss_pv = self.alpha * KL_loss(neg_prediction_pv_pretrain, neg_prediction_target)

                pos_loss_pv = torch.mean(kl_neg_loss_rest - denoise_pos_loss_pv)
                neg_loss_pv = torch.mean(kl_neg_loss_both - denoise_neg_loss_pv)

                neg_loss_buy_rest = torch.mean(kl_pos_loss_pv - denoise_neg_loss_buy_rest)
                neg_loss_buy_both = torch.mean(kl_neg_loss_pv - denoise_neg_loss_buy_both)

                loss = pos_loss_pv + neg_loss_pv + neg_loss_buy_both + neg_loss_buy_rest

                reg_loss = 0
                for param in target_model.parameters():
                    reg_loss += param.norm(2).pow(2)

                reg_loss = 1 / 2 * reg_loss / float(self.user_num)

                loss += reg_loss * self.param['lambda1']
                loss.backward()
                _on_iteration_end()

                if count % 200 == 0 and count != 0:
                    print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))
                count += 1

            print("################### BUY TARGET + PRETRAIN ######################")
            recall, NDCG = evaluate.test_all_users_with_two_model(target_model, buy_pre_model, 4096,
                                                                  test_data_pos, user_pos, self.top_k,
                                                                  device=self.device,
                                                                  beta1=self.beta1, beta2=self.beta2)
            final_perf = "Iter=[%d]\t recall=[%s], ndcg=[%s]" % \
                         (epoch,
                          '\t'.join(['%.4f' % r for r in recall]),
                          '\t'.join(['%.4f' % r for r in NDCG]))
            print(final_perf)

            both_rec_loger.append(recall)
            both_ndcg_loger.append(NDCG)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(recall[-1], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        early_stop_rounds=self.early_stop_rounds)

            if should_stop:
                print("one-way-train early stop.")
                break

            if cur_best_pre_0 == recall[-1]:
                print(f"saving {model_name} model.")
                model_save_path = os.path.join(self.param['folder'],
                                               f"{self.param['dataset']}_target_{model_name}_{self.seed}_{str(self.param['lambda1'])}.pt")
                torch.save(target_model.state_dict(), model_save_path)

        both_recs = np.array(both_rec_loger)
        both_ndcgs = np.array(both_ndcg_loger)

        both_best_rec_0 = max(both_recs[:, -1])
        both_idx = list(both_recs[:, -1]).index(both_best_rec_0)

        final_perf = "Best Iter=[%d]\t recall=[%s], ndcg=[%s]" % \
                     (both_idx,
                      '\t'.join(['%.5f' % r for r in both_recs[both_idx]]),
                      '\t'.join(['%.5f' % r for r in both_ndcgs[both_idx]]))
        print(final_perf)
