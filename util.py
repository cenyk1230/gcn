from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os
import math
import cPickle as cp
import scipy.sparse as sp
#import _pickle as cp  # python3 compatability
import networkx as nx

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-num_epochs', type=int, default=100, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=str, default='32', help='dimension(s) of latent layers')
cmd_opt.add_argument('-slice_k', type=float, default=0.6, help='number of nodes kept after Slicing')
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='s2v output size')
cmd_opt.add_argument('-hidden', type=int, default=128, help='dimension of classification')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-dropout', type=bool, default=False, help='whether add dropout after dense layer')
cmd_opt.add_argument('-early_stop', type=int, default=-1, help='early stop threshold')
cmd_opt.add_argument('-dim_u', type=int, default=128, help='dim_u')
cmd_opt.add_argument('-dim_a', type=int, default=64, help='dim_a')
cmd_opt.add_argument('-dim_r', type=int, default=20, help='dim_r')
cmd_opt.add_argument('-dim_p', type=int, default=128, help='dim_p')
cmd_opt.add_argument('-gpu', type=str, default='1', help='gpu number')
cmd_opt.add_argument('-field_size', type=int, default=5, help='receptive field size')
cmd_opt.add_argument('-use_deg', type=int, default=0, help='whether to add degree feature')

cmd_args, _ = cmd_opt.parse_known_args()

cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]

print(cmd_args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = sp.coo_matrix(mx)
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model"""
    # degs = np.sum(adj, axis=1)
    # nodes = range(adj.shape[0])
    # nodes.sort(key=lambda x:degs[x], reverse=True)
    # adj = adj[nodes, :][:, nodes]
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.g = g
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)

        # self.adj = preprocess_adj(nx.adjacency_matrix(g))
        self.adj = nx.adjacency_matrix(g)

    def receptive_field(self, node, k):
        fields = {node: (0, -self.g.degree[node])}
        dis = 0
        while len(fields) < k:
            tmp = []
            dis += 1
            for i in fields:
                for key in self.g.neighbors(i):
                    if key not in fields:
                        tmp.append(key)
            if not tmp:
                break
            for key in tmp:
                fields[key] = (dis, -self.g.degree[key])
        fields = fields.items()
        fields.sort(key=lambda x:x[1])

        if len(fields) < k:
            return [node] * (k - len(fields)) + [x[0] for x in fields]

        return [fields[i][0] for i in range(k)]

    def preprocessing(self, k, max_d):
        encoder = OneHotEncoder(cmd_args.feat_dim)
        tags = [[x] for x in self.node_tags]
        tags = encoder.fit_transform(tags).toarray()
        if not self.node_features:
            self.node_features = tags
        else:
            self.node_features = np.concatenate([tags, node_features], axis=1)

        # if self.num_nodes < k:
        #     delta = k - self.num_nodes
        #     self.g.add_nodes_from(list(range(self.num_nodes, k)))
        #     self.num_nodes = k
        #     self.node_features = np.concatenate([self.node_features, np.zeros((delta, cmd_args.feat_dim))], axis=0)
        #     self.adj = sp.csr_matrix((self.adj.data, self.adj.indices, list(self.adj.indptr) + [len(self.adj.data)] * delta), shape=(k, k))

        # if self.num_nodes > k:
        #     suf = list(range(self.num_nodes))
        #     random.shuffle(suf)
        #     self.node_features = self.node_features[suf[:k], :]
        #     self.num_nodes = k
        #     self.g = self.g.subgraph(suf[:k])
        #     self.g = nx.relabel_nodes(self.g, dict(zip(self.g.nodes(), list(range(k)))))
        #     self.adj = nx.adjacency_matrix(self.g)

        if cmd_args.use_deg == 1:
            degs = [[self.g.degree[i]] for i in range(self.num_nodes)]
            encoder_deg = OneHotEncoder(max_d + 1)
            degs = encoder_deg.fit_transform(degs).toarray()
            self.node_features = np.concatenate([self.node_features, degs], axis=1)

        # tmp = []
        # for i in range(self.num_nodes):
        #     # tmp_adj = self.adj[i].indices
        #     # if len(tmp_adj) < cmd_args.field_size:
        #     #     tmp.append([i] * (cmd_args.field_size - len(tmp_adj)) + list(tmp_adj))
        #     # else:
        #     #     tmp.append([i] + list(np.random.choice(tmp_adj, cmd_args.field_size - 1)))
        #     tmp.append(self.receptive_field(i, cmd_args.field_size))

        # self.adj = np.array(tmp)

        if self.num_nodes < k:
            delta = k - self.num_nodes
            self.num_nodes = k
            self.node_features = np.concatenate([self.node_features, np.zeros((delta, self.node_features.shape[1]))], axis=0)
            # self.adj = (self.adj[0], self.adj[1], (k, k))
            self.adj = preprocess_adj(self.adj)
            self.adj = (self.adj[0], self.adj[1], (k, k))
        elif self.num_nodes > k:
            suf = list(range(self.num_nodes))
            random.shuffle(suf)
            self.node_features = self.node_features[suf[:k], :]
            self.adj = self.adj[suf[:k], :][:, suf[:k]]
            self.adj = preprocess_adj(self.adj)
        else:
            self.adj = preprocess_adj(self.adj)
        

def load_data():
    print('loading data')

    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('data/%s/%s.txt' % (cmd_args.data, cmd_args.data), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                row = [int(w) for w in row]
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
            #assert len(g.edges()) * 2 == n_edges  (some graphs in COLLAB have self-loops, ignored here)
            assert len(g) == n
            g_list.append(S2VGraph(g, l, node_tags))
    for g in g_list:
        g.label = label_dict[g.label]
    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim = len(feat_dict)

    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)

    if cmd_args.slice_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in g_list])
        cmd_args.slice_k = num_nodes_list[int(math.ceil(cmd_args.slice_k * len(num_nodes_list))) - 1]
        print('k used in Slicing is: ' + str(cmd_args.slice_k))

    max_d = 0
    for g in g_list:
        degs = list(g.g.degree)
        tmp_d = max([x[1] for x in degs])
        if tmp_d > max_d:
            max_d = tmp_d

    for g in g_list:
        g.preprocessing(cmd_args.slice_k, max_d)

    if cmd_args.use_deg == 1:
        cmd_args.feat_dim += max_d + 1

    train_idxes = np.loadtxt('data/%s/10fold_idx/train_idx-%d.txt' % (cmd_args.data, cmd_args.fold), dtype=np.int32).tolist()
    test_idxes = np.loadtxt('data/%s/10fold_idx/test_idx-%d.txt' % (cmd_args.data, cmd_args.fold), dtype=np.int32).tolist()

    # train_idxes, val_idxes = train_test_split(train_idxes, test_size=0.1, random_state=cmd_args.seed)

    return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes]

