#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
Tăng cường danh mục ẩn Anchor-Protected Category-Guided Augmentation và bảo vệ id sản phẩm cuối cùng ko cho phép che giấu
"""

import argparse
import pickle
import time
import logging
from datetime import datetime
from utils import build_graph, Data, split_validation
from model import trans_to_cuda, SessionGraph, train_test

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')

# ===================================================================
# THÊM CHO MODULE 2: CÁC THAM SỐ CONTRASTIVE LEARNING (CL)
# ===================================================================
parser.add_argument('--lambda_cl', type=float, default=0.1, help='Trọng số của InfoNCE Loss (Thường 0.01 đến 0.1)')
parser.add_argument('--aug_p', type=float, default=0.2, help='Tỷ lệ tráo đổi item trong phiên (Same-leaf substitution ratio)')
parser.add_argument('--tau', type=float, default=0.1, help='Nhiệt độ (Temperature) cho InfoNCE Loss')
# ===================================================================

opt = parser.parse_args()

# Cấu hình logging
log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Arguments: {opt}")

def main():
    # 1. Load từ điển category để lấy tự động n_categories
    try:
        # Lưu ý: Nhớ đổi đường dẫn này trỏ đúng vào thư mục code thực tế của bạn
        cat_dict = pickle.load(open('/Users/admin/Documents/tài liệu tham khảo NCS/NCS/SR-GNN_nang_cap_modun2/datasets/cat_dict.pkl', 'rb')) 
        n_categories = len(cat_dict) + 1 
    except FileNotFoundError:
        logger.warning("Không tìm thấy cat_dict.pkl. Mặc định n_categories = 0")
        n_categories = 0

    # ===================================================================
    # THÊM CHO MODULE 2: LOAD TỪ ĐIỂN TRA CỨU AUGMENTATION
    # ===================================================================
    try:
        item2cat = pickle.load(open('/Users/admin/Documents/tài liệu tham khảo NCS/NCS/SR-GNN_nang_cap_modun2/datasets/item2cat_internal.pkl', 'rb'))
        cat2items = pickle.load(open('/Users/admin/Documents/tài liệu tham khảo NCS/NCS/SR-GNN_nang_cap_modun2/datasets/cat2items_dict.pkl', 'rb'))
        logger.info(f"Đã load thành công từ điển Augmentation: {len(cat2items)} danh mục khả dụng.")
    except FileNotFoundError:
        logger.warning("Không tìm thấy file từ điển (item2cat_internal.pkl hoặc cat2items_dict.pkl) cho Module 2!")
        item2cat, cat2items = {}, {}
    # ===================================================================

    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))

    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))

    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    opt.n_categories = n_categories 
    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    
    for epoch in range(opt.epoch):
        logger.info('-------------------------------------------------------')
        logger.info(f'epoch: {epoch}')
        
        # ===================================================================
        # SỬA CHO MODULE 2: TRUYỀN TỪ ĐIỂN VÀ OPT VÀO HÀM TRAIN
        # ===================================================================
        hit, mrr = train_test(model, train_data, test_data, item2cat, cat2items, opt)
        # ===================================================================
        
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        logger.info('Best Result:')
        logger.info(f'\tRecall@20:\t{best_result[0]:.4f}\tMMR@20:\t{best_result[1]:.4f}\tEpoch:\t{best_epoch[0]},\t{best_epoch[1]}')
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
            
    logger.info('-------------------------------------------------------')
    end = time.time()
    logger.info(f"Run time: {end - start:.6f} s")
    logger.info(f"Log saved to: {log_filename}")

if __name__ == '__main__':
    main()