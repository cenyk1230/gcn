import numpy as np
import tensorflow as tf

from util import cmd_args

np.random.seed(cmd_args.seed)
tf.set_random_seed(cmd_args.seed)

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32, seed=cmd_args.seed)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging

    def _call(self, inputs):
        raise NotImplementedError

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=tf.nn.relu, bias=False, **kwargs):
        super(Dense, self).__init__(**kwargs)


        self.dropout = dropout
        self.act = act
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.dropout > 0:
            x = tf.nn.dropout(x, 1 - self.dropout, seed=cmd_args.seed)

        # transform
        output = dot(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, field_size, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.act = act
        self.bias = bias
        self.field_size = field_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim * field_size, output_dim],
                                      name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        adjs, x = inputs

        # convolve
        output = []
        for i in range(len(adjs)):
            tmp_x = tf.gather(x[i], adjs[i])
            tmp_x = tf.reshape(tmp_x, [-1, self.input_dim * self.field_size])
            if self.dropout > 0:
                # x[i] = tf.nn.dropout(x[i], 1 - self.dropout, seed=cmd_args.seed)
                tmp_x = tf.nn.dropout(tmp_x, 1 - self.dropout, seed=cmd_args.seed)
            # tmp = dot(adjs[i], dot(x[i], self.vars['weights']), sparse=self.sparse_inputs)
            tmp = dot(tmp_x, self.vars['weights'])
            if self.bias:
                tmp += self.vars['bias']
            tmp = self.act(tmp)
            output.append(tmp)

        return output
