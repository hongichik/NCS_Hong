#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os

# Khai báo tham số đầu vào, hiện tại chỉ có 1 file csv nên không thể sử dụng các phương án khác
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
elif opt.dataset =='yoochoose':
    dataset = 'yoochoose/yoochoose-clicks.dat'


print("-- Starting @ %ss" % datetime.datetime.now())

# Mảng 1 chia dữ liệu thành các session và sắp xếp thời gian click của mỗi session
with open(dataset, "r") as f:
    # Đọc file csv, phân tách bằng dấu phẩy hoặc dấu chấm phẩy tùy thuộc vào dataset
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, delimiter=',', fieldnames=['session_id', 'timestamp', 'item_id', 'category']) # Thêm: Khai báo fieldnames
    else:
        reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}
    
    # Thêm: Khởi tạo Từ điển trực tiếp để hứng Category
    item2cat_raw = {}
    
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        # Thêm: Bỏ qua dòng Header nếu có
        if data['session_id'] == 'session_id':
            continue
            
        sessid = data['session_id']
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid

        if opt.dataset == 'yoochoose':
            item = data['item_id']
            # Thêm: Trực tiếp lấy Category, xử lý rác 'S' và '0'
            raw_cat = data['category'].strip()
            if raw_cat != 'S' and raw_cat != '0':
                 item2cat_raw[int(item)] = int(raw_cat)
        else:
            # Lấy ra món đồ chọn vào và thời gian chọn chọn để có thể sắp xếp ở mục sau
            item = data['item_id'], int(data['timeframe'])

        # Thời gian của mỗi session được lấy từ trường 'timestamp' hoặc 'eventdate' tùy thuộc vào dataset
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        # Nếu chưa có session_id nào thì tạo mới, nếu đã có thì thêm item vào session đó
        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        # Chuyển đổi thời gian từ định dạng chuỗi sang định dạng timestamp để có thể so sánh và sắp xếp theo thời gian click
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            # hàm sắp xếp theo thời gian
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Thêm: Sau khi quét xong file, lưu cuốn từ điển này lại để dùng cho Module 2 (Contrastive Learning)
print("-- Saving internal item2cat mapping...")
pickle.dump(item2cat_raw, open('item2cat_internal.pkl', 'wb'))

# Loại bỏ các session có độ dài là 1 vì không có ý nghĩa dự đoán
# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Loại bỏ các item xuất hiện ít hơn 5 lần vì không có ý nghĩa dự đoán
# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# chia dữ liệu thành train và test, test lấy 7 ngày cuối, train là thời gian còn lại
# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
cat_dict = {} # Thêm: Từ điển đánh số thứ tự cho Category (bắt đầu từ 1)

# Chuyển đổi ID của session thành các chuỗi từ 1-n, không đánh từ 0 vì 0 được dùng để padding sau này
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_cats = [] # Thêm: Mảng lưu trữ chuỗi Category
    train_dates = []
    item_ctr = 1
    cat_ctr = 1 # Thêm: Bộ đếm để mã hóa ID category
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        outseq_cat = [] # Thêm: Chuỗi category song song với item
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
            
            # Thêm: Quá trình ánh xạ và mã hóa Category từ 1-N (Dùng trực tiếp từ điển nội bộ)
            raw_item = int(i) if str(i).isdigit() else i
            c = item2cat_raw.get(raw_item, 0)
            if c in cat_dict:
                outseq_cat += [cat_dict[c]]
            else:
                outseq_cat += [cat_ctr]
                cat_dict[c] = cat_ctr
                cat_ctr += 1

        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
        train_cats += [outseq_cat] # Thêm: Lưu vào mảng tổng
    print('Total items:', item_ctr)     # 43098, 37484
    print('Total categories:', cat_ctr) # Thêm: In ra để biết cần tạo Embedding size bao nhiêu
    
    # Thêm: Lưu cat_dict ra file để dùng khai báo nn.Embedding() trong model.py sau này
    pickle.dump(cat_dict, open('cat_dict.pkl', 'wb'))
    
    return train_ids, train_dates, train_seqs, train_cats # Thêm: return train_cats


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_cats = [] # Thêm: Mảng lưu trữ chuỗi Category cho test
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        outseq_cat = [] # Thêm: Chuỗi category
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
                
                # Thêm: Lấy ID category đã được mã hóa ở tập Train
                raw_item = int(i) if str(i).isdigit() else i
                c = item2cat_raw.get(raw_item, 0)
                outseq_cat += [cat_dict.get(c, 0)]

        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
        test_cats += [outseq_cat] # Thêm: Lưu vào mảng tổng
    return test_ids, test_dates, test_seqs, test_cats # Thêm: return test_cats


tra_ids, tra_dates, tra_seqs, tra_cats = obtian_tra() # Thêm: nhận biến tra_cats
tes_ids, tes_dates, tes_seqs, tes_cats = obtian_tes() # Thêm: nhận biến tes_cats

print("-------")

for i in range(10):
    print(tra_ids[i], tra_dates[i], tra_seqs[i], tra_cats[i]) # Thêm in ra tra_cats
print(len(tra_ids), len(tra_dates), len(tra_seqs))
# exit(0)


def process_seqs(iseqs, idates, icats): # Thêm: truyền biến icats vào hàm
    out_seqs = []
    out_cats = [] # Thêm: mảng cắt chuỗi category
    out_dates = []
    labs = []
    ids = []
    for id, seq, cat_seq, date in zip(range(len(iseqs)), iseqs, icats, idates): # Thêm: zip cat_seq
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_cats += [cat_seq[:-i]] # Thêm: cắt chuỗi category tương tự như cắt item
            out_dates += [date]
            ids += [id]
    return out_seqs, out_cats, out_dates, labs, ids # Thêm: return out_cats


# Thêm: Truyền tra_cats và tes_cats vào hàm
tr_seqs, tr_cats, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates, tra_cats)
te_seqs, te_cats, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates, tes_cats)
print(len(tr_seqs), len(tr_dates), len(tr_labs), len(tr_ids))
for i in range(10):
    print(tr_ids[i], tr_dates[i], tr_seqs[i], tr_cats[i], tr_labs[i]) # Thêm in ra tr_cats

tra = (tr_seqs, tr_cats, tr_labs) # Thêm: Đóng gói thành Tuple 3 phần tử
tes = (te_seqs, te_cats, te_labs) # Thêm: Đóng gói thành Tuple 3 phần tử
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))
if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    # Thêm: Chia dữ liệu Tuple 3 phần tử
    tra4 = (tr_seqs[-split4:], tr_cats[-split4:], tr_labs[-split4:])
    tra64 = (tr_seqs[-split64:], tr_cats[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))

else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')