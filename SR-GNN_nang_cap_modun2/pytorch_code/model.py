#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import logging

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.n_categories = opt.n_categories 
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.cat_embedding = nn.Embedding(self.n_categories, self.hidden_size) 
        
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1] 
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1]) 
        q2 = self.linear_two(hidden) 
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:] 
        scores = torch.matmul(a, b.transpose(1, 0))
        
        # ================== THÊM CHO MODULE 2 ==================
        # Trả về thêm `a` (Session Embedding) để tính Contrastive Loss
        return scores, a 
        # =======================================================

    def forward(self, inputs, cats, A): 
        hidden_items = self.embedding(inputs) 
        hidden_cats = self.cat_embedding(cats) 
        hidden = hidden_items + hidden_cats 
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

# ================== THÊM CHO MODULE 2 ==================
def augment_items(items, item2cat, cat2items, p=0.2):
    """ Hàm Node Feature Substitution: Tráo đổi ngẫu nhiên item cùng danh mục (Bảo vệ Anchor Item) """
    items_aug = np.copy(items)
    for row in range(items_aug.shape[0]):
        # Tìm các vị trí có item thực sự (bỏ qua số 0 padding)
        valid_indices = np.where(items_aug[row] != 0)[0]
        
        # SỬA LỖI CHÍ MẠNG: Bỏ qua item cuối cùng (Anchor Item - Local Interest)
        if len(valid_indices) <= 1:
            continue # Nếu phiên chỉ có 1 món, không tráo đổi gì cả
            
        # Loại bỏ vị trí cuối cùng ra khỏi danh sách được phép tráo đổi
        valid_indices = valid_indices[:-1] 
        
        num_to_sub = max(1, int(len(valid_indices) * p))
        sub_indices = random.sample(list(valid_indices), min(num_to_sub, len(valid_indices)))
        
        for idx in sub_indices:
            old_item = items_aug[row][idx]
            cat = item2cat.get(old_item, 0)
            if cat in cat2items and len(cat2items[cat]) > 1:
                candidate_items = [i for i in cat2items[cat] if i != old_item]
                if candidate_items:
                    items_aug[row][idx] = random.choice(candidate_items)
    return items_aug
# =======================================================

def forward(model, i, data, is_train=False, item2cat=None, cat2items=None, opt=None, is_aug=False):
    alias_inputs, A, items, cats, mask, targets = data.get_slice(i) 
    
    # ================== THÊM CHO MODULE 2 ==================
    if is_train and is_aug:
        items = augment_items(items, item2cat, cat2items, p=opt.aug_p)
    # =======================================================
    
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    cats = trans_to_cuda(torch.Tensor(cats).long()) 
    A = trans_to_cuda(torch.Tensor(np.array(A)).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    
    hidden = model(items, cats, A) 
    
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    
    # ================== THÊM CHO MODULE 2 ==================
    # Hứng thêm z_session (vector đại diện của cả phiên)
    scores, z_session = model.compute_scores(seq_hidden, mask)
    return targets, scores, z_session
    # =======================================================


def train_test(model, train_data, test_data, item2cat=None, cat2items=None, opt=None):
    logger = logging.getLogger(__name__)
    logger.info(f'start training: {datetime.datetime.now()}')
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        
        # 1. FORWARD GỐC (Main Task)
        targets, scores, z_orig = forward(model, i, train_data, is_train=True, is_aug=False)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        
        # Loss 1: Cross-Entropy cho bài toán dự đoán (Recommendation Loss)
        loss_rec = model.loss_function(scores, targets - 1)
        
        # ================== THÊM CHO MODULE 2 (CONTRASTIVE LEARNING) ==================
        loss = loss_rec
        
        if opt is not None and item2cat is not None:
            # 2. FORWARD ẢO (Augmented Task)
            _, _, z_aug = forward(model, i, train_data, is_train=True, 
                                  item2cat=item2cat, cat2items=cat2items, opt=opt, is_aug=True)
            
            # 3. TÍNH INFO-NCE LOSS
            # Chuẩn hóa vector lên hình cầu đơn vị (Hypersphere)
            z_orig_norm = F.normalize(z_orig, dim=1)
            z_aug_norm = F.normalize(z_aug, dim=1)
            
            # Tính ma trận độ tương đồng Cosine chia cho nhiệt độ tau
            sim_matrix = torch.matmul(z_orig_norm, z_aug_norm.T) / opt.tau
            
            # Khai báo nhãn: Dữ liệu trên đường chéo chính (Positive pairs) là 1, còn lại là 0 (Negative)
            cl_labels = trans_to_cuda(torch.arange(z_orig.size(0)).long())
            
            loss_cl = F.cross_entropy(sim_matrix, cl_labels)
            
            # Tổng hợp Loss
            loss = loss_rec + opt.lambda_cl * loss_cl
        # ==============================================================================
        
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        
        if j % int(len(slices) / 5 + 1) == 0:
            logger.info(f'[{j}/{len(slices)}] Loss: {loss.item():.4f}')
            
    logger.info(f'\tLoss:\t{total_loss:.3f}')

    logger.info(f'start predicting: {datetime.datetime.now()}')
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        # Ở bước Test, ta không dùng augmentation, bỏ qua z_session
        targets, scores, _ = forward(model, i, test_data, is_train=False)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    model.scheduler.step()
    return hit, mrr