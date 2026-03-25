#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
import logging
from datetime import datetime
from utils import build_graph, Data, split_validation
from model import trans_to_cuda, SessionGraph, train_test

# Cấu hình nhằm mục đích dễ dàng thay đổi các tham số khi chạy chương trình, ví dụ như dataset, batch size, hidden size, số epoch, learning rate, v.v. Các tham số này sẽ được truyền vào khi chạy chương trình từ command line.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
# Số session được đưa vào mỗi lần chạy, đang để là 100
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
# Kích thước của vector giạng ẩn trong mô hình, mỗi giá trị sẽ được biểu diễn thành vector có số lượng đã cài đặt, nếu giá trị thấp quá dự đoán kém, giá trị cao quá dễ bị overfitting.
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
# Số lần lặp lại tối đa khi huấn luyện mô hình, đang để là 30
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
# Tốc độ học ban đầu (đang để tốc độ học ban đầu khá cao)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
# Thời gian tăng lên khi đạt đủ số lần cần thiết
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
# Cứ 3 vòng tốc độ sẽ tăng thêm --lr_dc đảm bảo việc học về sau được tối ưu hơn
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
# Hàm phạt giúp cho hệ thống không bị overfitting, giá trị càng cao thì hệ thống càng bị phạt nặng khi có trọng số lớn, giá trị thấp quá thì hệ thống dễ bị overfitting, giá trị cao quá thì hệ thống khó học được, đang để là 1e-5
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
# Số bước lan truyền trong GNN, giá trị là 1 tức là B có thể nhìn thấy A và C, A chỉ có thể nhìn thấy B mà không thấy C, trong bài toán này vì dữ liệu mua sắp quá ít để 1 là ổn định nhất
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
# Tắt thì thuật toán chỉ đánh giá cục bộ bật thuật toán đánh giá toàn cục
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
# Hệ thống trích ra 1 phần để làm bài test, nếu bật thì thuật toán sẽ đánh giá trên tập validation, nếu tắt thì thuật toán sẽ đánh giá trên tập test, mặc định là tắt
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()

# Cấu hình logging để ghi vào file
log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Vẫn hiển thị trên console nếu muốn
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Arguments: {opt}")

def main():
    # Thêm: Load từ điển category để lấy tự động n_categories
    try:
        cat_dict = pickle.load(open('/Users/admin/Documents/tài liệu tham khảo NCS/NCS/SR-GNN_nang_cap/datasets/cat_dict.pkl', 'rb'))
        n_categories = len(cat_dict) + 1 # Cộng 1 vì PyTorch embedding thường bắt đầu từ index 1 (0 dùng để padding)
    except FileNotFoundError:
        logger.warning("Không tìm thấy cat_dict.pkl. Mặc định n_categories = 0")
        n_categories = 0

    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    # Lúc này train_data là Tuple 3 phần tử: (train_seqs, train_cats, train_labs)

    if opt.validation:
        # Bạn nhớ phải sang file utils.py sửa lại hàm split_validation để nó nhận 3 mảng nhé
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))

    # Hàm Data() bên utils.py cũng cần được sửa để nhận 3 mảng
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)

    # Khai báo tự động số node thay vì hard-code (Lấy Max ID từ tập Train)
    # Vì file preprocess.py đánh số từ 1, nên số lượng node = max_id + 1
    # Ở đây mình vẫn giữ nguyên logic cũ của bạn để không phá vỡ Baseline
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    # Thêm n_categories vào SessionGraph
    # (Bạn sẽ cần vào file model.py để khai báo thêm biến n_categories này ở hàm __init__ của SessionGraph)
    # Tạm thời ở bản này, để code không báo lỗi model.py, mình vẫn gọi như cũ, hoặc truyền vào qua opt.
    opt.n_categories = n_categories 
    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        logger.info('-------------------------------------------------------')
        logger.info(f'epoch: {epoch}')
        # Hàm train_test() bên model.py sẽ tự động bốc dữ liệu từ object Data đã cập nhật
        hit, mrr = train_test(model, train_data, test_data)
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