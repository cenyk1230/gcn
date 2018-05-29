from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pickle
import random
import math
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from layers import GraphConvolution, Dense, glorot, zeros

from util import cmd_args, load_data

os.environ['CUDA_VISIBLE_DEVICES'] = cmd_args.gpu

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
tf.set_random_seed(cmd_args.seed)


def get_batches(graphs, batch_size):
    n_batches = (len(graphs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, s, m = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(graphs):
                index = index - len(graphs) * 2
                m.append(False)
            else:
                m.append(True)
            x.append(graphs[index].node_features)
            y.append(graphs[index].label)
            s.append(graphs[index].adj)
        yield x, y, s, m

train_graphs, test_graphs = load_data()

print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

k = cmd_args.slice_k

feat_dim = cmd_args.feat_dim

num_class = cmd_args.num_class

batch_size = cmd_args.batch_size

dim_u = cmd_args.dim_u

dim_a = cmd_args.dim_a

dim_r = cmd_args.dim_r

dim_p = cmd_args.dim_p

latent_dim = cmd_args.latent_dim

hidden = cmd_args.hidden

field_size = cmd_args.field_size

tf_graph = tf.Graph()

with tf_graph.as_default():
    adjs = [tf.sparse_placeholder(tf.float32, shape=[None, None]) for _ in range(batch_size)]
    # adjs = [tf.placeholder(tf.int32, shape=[None, field_size]) for _ in range(batch_size)]
    features = [tf.placeholder(tf.float32, shape=[None, feat_dim]) for _ in range(batch_size)]
    # adjs = tf.placeholder(tf.float32, shape=[batch_size, k, k])
    # features = tf.placeholder(tf.float32, shape=[batch_size, k, feat_dim])
    labels = tf.placeholder(tf.int32, shape=[batch_size, num_class], name='labels')
    mask = tf.placeholder(tf.bool, shape=[batch_size])

with tf_graph.as_default():

    gc_layers = []

    GC1 = GraphConvolution(input_dim=feat_dim, output_dim=latent_dim, k=k, dropout=0.0, sparse_inputs=True, act=tf.nn.relu, bias=True)
    GC2 = GraphConvolution(input_dim=latent_dim, output_dim=latent_dim, k=k, dropout=0.0, sparse_inputs=True, act=tf.nn.relu, bias=True)
    GC3 = GraphConvolution(input_dim=latent_dim, output_dim=latent_dim, k=k, dropout=0.0, sparse_inputs=True, act=tf.nn.relu, bias=True)
    GC4 = GraphConvolution(input_dim=latent_dim, output_dim=latent_dim, k=k, dropout=0.0, sparse_inputs=True, act=tf.nn.relu, bias=True)

    # GC1 = GraphConvolution(input_dim=12, output_dim=8, act=tf.nn.relu, bias=False)
    # GC2 = GraphConvolution(input_dim=8, output_dim=2, act=lambda x:x, bias=False)

    # gc_layers.append(GC1)
    # gc_layers.append(GC2)
    # gc_layers.append(GC3)
    # gc_layers.append(GC4)
    # gc_layers.append(GC5)

    lstm_fw = tf.contrib.rnn.LSTMCell(dim_u)
    lstm_bw = tf.contrib.rnn.LSTMCell(dim_u)

    Ws1 = glorot((dim_a, dim_u * 2))
    Ws2 = glorot((dim_r, dim_a))

    I = tf.stack([tf.eye(dim_r) for _ in range(batch_size)])

    W_pool = glorot((latent_dim*3, dim_p))
    b_pool = zeros((dim_p))

    # latent_dim*3+1   dim_u*2
    Conv1 = tf.layers.Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    Maxp1 = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
    Conv2 = tf.layers.Conv1D(filters=32, kernel_size=4, padding='same', activation=tf.nn.relu)

    # Conv1 = tf.layers.Conv1D(filters=128, kernel_size=4, padding='same', activation=tf.nn.relu)
    # Maxp1 = tf.layers.MaxPooling1D(pool_size=2, strides=2)

    # hidden_state = tf.zeros([batch_size, 32])
    # current_state = tf.zeros([batch_size, 32])

    fc_layers = []

    # dim_r*dim_u*2  k*dim_u  dim_p*16
    # FC1 = Dense(input_dim=(latent_dim*3+1)*k*16, output_dim=128, dropout=0.5, act=tf.nn.relu, bias=True)
    # FC2 = Dense(input_dim=16, output_dim=16, act=tf.nn.relu, bias=True)
    # FC3 = Dense(input_dim=16, output_dim=8, act=tf.nn.relu, bias=True)
    # FC4 = Dense(input_dim=128, output_dim=num_class, act=lambda x:x, bias=True)

    # FC1 = tf.layers.Dense(16, activation=tf.nn.relu, use_bias=True)
    # FC2 = tf.layers.Dense(16, activation=tf.nn.relu, use_bias=True)
    # FC3 = tf.layers.Dense(8, activation=tf.nn.relu, use_bias=True)
    # FC4 = tf.layers.Dense(2, activation=lambda x:x, use_bias=True)

    # fc_layers.append(FC1)
    # fc_layers.append(FC2)
    # fc_layers.append(FC3)
    # fc_layers.append(FC4)
    # fc_layers.append(FC5)

    Dense1 = tf.layers.Dense(hidden, activation=tf.nn.relu, use_bias=True)
    Dense2 = tf.layers.Dense(num_class, use_bias=True)

    Dropout1 = tf.keras.layers.Dropout(rate=0.5, seed=cmd_args.seed)


    LSTM_layer = tf.keras.layers.LSTM(units=dim_u)

with tf_graph.as_default():
    # X_gc = features
    # for GC in gc_layers:
    #     X_gc = GC((adjs, X_gc))
    X_gc1 = GC1((adjs, features))
    X_gc2 = GC2((adjs, X_gc1))
    X_gc3 = GC3((adjs, X_gc2))
    X_gc4 = GC4((adjs, X_gc3))

    X_gc = []
    for i in range(batch_size):
        X_gc.append(tf.concat([X_gc1[i], X_gc2[i], X_gc3[i], X_gc4[i]], axis=1))
        # X_gc.append(tf.concat([X_gc1[i], X_gc2[i], X_gc3[i]], axis=1))
        # X_gc.append(tf.concat([X_gc1[i], X_gc2[i]], axis=1))

    # X_n = []
    # for i in range(batch_size):
        # tmp1 = tf.reshape(tf.slice(X[i], [0, 0], [k, 16]), [-1])
        # tmp2 = tf.reduce_mean(X[i], axis=0)
        # tmp3 = tf.reduce_min(X[i], axis=0)
        # tmp4 = tf.reduce_max(X[i], axis=0)

        # tmp = tf.concat([tmp1, tmp2, tmp3, tmp4], axis=0)

        # X_n.append(tf.slice(X_gc[i], [0, 0], [k, latent_dim*3+1]))
        
        # indices = tf.nn.top_k(tf.reshape(X_gc4[i], [-1]), k).indices
        # X_topk = tf.gather(X_gc[i], indices)
        
        # X_n.append(tf.reshape(X_topk, [-1]))
        # X_n.append(X_topk)

        # X_pool = tf.reduce_max(tf.nn.relu(tf.matmul(X_gc[i], W_pool) + b_pool), axis=0)

        # X_n.append(tf.concat([X_pool, tf.reduce_max(X_gc[i], axis=0)], axis=0))
        # X_n.append(X_pool)
        # X_n.append(X_gc[i])

    # X_lstm = tf.stack(X_n)
    # X_n = X_lstm

    # X_lstm = Conv1(X_lstm)
    # X_lstm = Maxp1(X_lstm)

    # X_lstm = tf.unstack(X_lstm, axis=1)

    # H, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw, lstm_bw, X_lstm, dtype=tf.float32)

    # print(len(H), H[0].shape)

    # H = LSTM_layer(X_lstm)

    # H = tf.reshape(tf.transpose(tf.stack(H), [1, 0, 2]), [batch_size, -1])

    # X_n = tf.expand_dims(tf.stack(X_n), -1)
    # X_n = tf.expand_dims(H, -1)

    X_n = tf.stack(X_gc)
    X_n = Conv1(X_n)
    X_n = Maxp1(X_n)
    X_n = Conv2(X_n)
    # X_n = H
    
    # (k+1)//2 * 32
    outputs_conv = tf.reshape(X_n, [batch_size, (k+1)//2 * 32])

    outputs_hidden = Dense1(outputs_conv)
    outputs_dropout = Dropout1(outputs_hidden)
    outputs = Dense2(outputs_dropout)

    # for FC in fc_layers:
    #     outputs = FC(outputs)
    

    pred = tf.equal(tf.argmax(labels, 1), tf.argmax(outputs, 1))

    pred = tf.boolean_mask(pred, mask)

    pred = tf.reduce_mean(tf.cast(pred, tf.float32))

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outputs)

    # P = tf.matmul(A, tf.transpose(A, [0, 2, 1])) - I
    
    # loss_p = tf.norm(P, ord='fro', axis=(1, 2))

    cost = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=cmd_args.learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=cmd_args.learning_rate).minimize(cost)


def generate_adjs(s):
    ret = np.zeros((batch_size, k, k))
    indices = []
    values = []
    shape = (batch_size, k, k)
    for i in range(len(s)):
        coords, v, _ = s[i]
        for j in range(len(coords)):
            indices.append((i, coords[j][0], coords[j][1]))
            values.append(v[j])
            ret[i, coords[j][0], coords[j][1]] = 1.0
    # return (indices, values, shape)
    return ret

def evaluate(graphs):
    batches = get_batches(graphs, batch_size)
    loss = []
    accu = []
    for x, y, s, m in batches:
        tmp_label = np.zeros((len(y), num_class), dtype=np.int32)
        for i in range(len(y)):
            tmp_label[i, y[i]] = 1

        feed = {labels: tmp_label}
        feed.update({adjs[i]: s[i] for i in range(len(s))})
        feed.update({features[i]: x[i] for i in range(len(x))})
        # feed.update({adjs: generate_adjs(s)})
        # feed.update({features: x})
        # feed.update({sizes[i]: len(s[i]) for i in range(len(s))})
        feed.update({mask: m})
        feed.update({tf.keras.backend.learning_phase(): 0})

        cur_loss, cur_accu = sess.run([cost, pred], feed_dict = feed)

        loss.extend([cur_loss] * np.sum(m))
        accu.extend([cur_accu] * np.sum(m))

    return np.mean(loss), np.mean(accu)

def get_hidden(graphs):
    embeddings = []
    label_list = []
    batches = get_batches(graphs, batch_size)
    for x, y, s, m in batches:
        tmp_label = np.zeros((len(y), num_class), dtype=np.int32)
        for i in range(len(y)):
            tmp_label[i, y[i]] = 1

        feed = {labels: tmp_label}
        feed.update({adjs[i]: s[i] for i in range(len(s))})
        feed.update({features[i]: x[i] for i in range(len(x))})
        feed.update({mask: m})
        feed.update({tf.keras.backend.learning_phase(): 0})

        hiddens = sess.run([outputs_hidden], feed_dict = feed)

        embeddings.append(np.array(hiddens[0])[m])
        label_list.append(np.array(y)[m])
    
    embeddings = np.concatenate(embeddings, axis=0)
    label_list = np.concatenate(label_list, axis=0)

    return (embeddings, label_list)

epochs = cmd_args.num_epochs

with tf.Session(graph=tf_graph) as sess:

    loss = []
    accu = []
    sess.run(tf.global_variables_initializer())

    max_accu = 0

    f = open('accu.txt', 'w')

    logs = np.zeros([epochs, 5])

    for e in range(1, epochs+1):
        random.shuffle(train_graphs)
        batches = get_batches(train_graphs, batch_size)
        start = time.time()

        loss_e = []
        accu_e = []

        for x, y, s, m in batches:
            tmp_label = np.zeros((len(y), num_class), dtype=np.int32)
            for i in range(len(y)):
                tmp_label[i, y[i]] = 1

            feed = {labels: tmp_label}
            feed.update({adjs[i]: s[i] for i in range(len(s))})
            feed.update({features[i]: x[i] for i in range(len(x))})
            # feed.update({adjs: generate_adjs(s)})
            # feed.update({features: x})
            # feed.update({sizes[i]: len(s[i]) for i in range(len(s))})
            feed.update({mask: m})
            feed.update({tf.keras.backend.learning_phase(): 1})

            train_loss, train_accu, _ = sess.run([cost, pred, optimizer], feed_dict=feed)
            
            loss_e.extend([train_loss] * np.sum(m))
            accu_e.extend([train_accu] * np.sum(m))

        test_loss, test_accu = evaluate(test_graphs)

        f.write(str(e) + ', test accu: ' + str(test_accu) + '\n')

        if test_accu > max_accu:
            max_accu = test_accu

        end = time.time()

        loss.append(np.mean(loss_e))
        accu.append(np.mean(accu_e))

        logs[e - 1, :] = [e, loss[-1], accu[-1], test_loss, test_accu]

        print("Epoch {}/{}".format(e, epochs),
              "{:.4f} sec".format((end-start)),
              "train loss: {:.4f}".format(loss[-1]),
              "train accu: {:.4f}".format(accu[-1]),
              "test loss: {:.4f}".format(test_loss),
              "test accu: {:.4f}".format(test_accu))

    f.close()

    with open('result_{}.txt'.format(cmd_args.seed), 'a') as f:
        f.write("{:.4f}\n".format(test_accu))

    embeddings_0, labels_0 = get_hidden(train_graphs)
    embeddings_1, labels_1 = get_hidden(test_graphs)
    embeddings = np.concatenate([embeddings_0, embeddings_1], axis=0)
    label_list = np.concatenate([labels_0, labels_1], axis=0)
    print(embeddings.shape)
    np.savetxt('graph_embedding.txt', embeddings, fmt='%.5f')
    np.savetxt('graph_label.txt', label_list, fmt='%d')

plt.figure(figsize = (8, 5))
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.plot(logs[:, 0], logs[:, 1], '-', label = 'training loss', color="blue", linewidth = 1)
plt.plot(logs[:, 0], logs[:, 3], '-', label = 'testing loss', color="green", linewidth = 1)
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)
# plt.title("Loss")
plt.legend(loc = 1, fontsize=16) 
plt.grid(color = '#95a5a6', linestyle = '-', linewidth = 1, axis = 'x', alpha = 0.5)
plt.grid(color = '#95a5a6', linestyle = '-', linewidth = 1, axis = 'y', alpha = 0.5)
plt.savefig("loss.pdf")

plt.figure(figsize = (8, 5))
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.plot(logs[:, 0], logs[:, 2], '-', label = 'training accuracy', color="blue", linewidth = 1)
plt.plot(logs[:, 0], logs[:, 4], '-', label = 'testing accuracy', color="green", linewidth = 1)
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
# plt.title("Accuracy")
plt.legend(loc = 1, fontsize=16) 
plt.grid(color = '#95a5a6', linestyle = '-', linewidth = 1, axis = 'x', alpha = 0.5)
plt.grid(color = '#95a5a6', linestyle = '-', linewidth = 1, axis = 'y', alpha = 0.5)
plt.savefig("accuracy.pdf")
