#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import networkx as nx
import numpy as np


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    # Sửa: Nhận 3 mảng thay vì 2
    train_set_x, train_set_c, train_set_y = train_set 
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    
    # Sửa: Chia đều cả mảng category (train_set_c)
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_c = [train_set_c[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_c = [train_set_c[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    # Sửa: Trả về Tuple 3 phần tử
    return (train_set_x, train_set_c, train_set_y), (valid_set_x, valid_set_c, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        categories = data[1] # Thêm: Hứng mảng Categories
        
        inputs, mask, len_max = data_masks(inputs, [0])
        
        # Thêm: Pad (nhét thêm số 0) vào mảng category cho dài bằng len_max giống như inputs
        us_cats = [upois + [0] * (len_max - len(upois)) for upois in categories]
        
        self.inputs = np.asarray(inputs)
        self.categories = np.asarray(us_cats) # Thêm: Lưu vào object
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[2]) # Sửa: Target giờ là phần tử thứ 3 (index 2)
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.categories = self.categories[shuffled_arg] # Thêm: Xáo trộn category đồng bộ với inputs
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        # Sửa: Lấy thêm categories
        inputs, categories, mask, targets = self.inputs[i], self.categories[i], self.mask[i], self.targets[i]
        
        items, cats, n_node, A, alias_inputs = [], [], [], [], [] # Thêm: cats array
        
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        
        # Sửa: Duyệt song song cả phiên sản phẩm và phiên danh mục
        for u_input, u_cat in zip(inputs, categories):
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            
            # Thêm: Ánh xạ 1-1 để tìm đúng Category cho các Unique Nodes
            cat_node = []
            for item in node:
                if item == 0:
                    cat_node.append(0) # Padding thì category cũng là 0
                else:
                    # Tìm vị trí đầu tiên item xuất hiện trong phiên, từ đó soi ra category của nó
                    idx = np.where(u_input == item)[0][0]
                    cat_node.append(u_cat[idx])
            # Pad category cho đủ max_n_node
            cats.append(cat_node + (max_n_node - len(node)) * [0])

            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
            
        # Sửa: Hàm giờ trả về thêm cats
        return alias_inputs, A, items, cats, mask, targets