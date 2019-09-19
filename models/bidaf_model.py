from base.base_model import BaseModel
import tensorflow as tf
import Skeleton


tf.flags.DEFINE_integer("batch_size", 128, "Minibatch size")
tf.flags.DEFINE_float("dropout", 0.5, "dropout")


FLAGS = tf.flags.FLAGS

class BiDAFModel(BaseModel, Skeleton):
    def __init__(self, config):
        super(BiDAFModel, self).__init__(config)

        self.build_model()
        self.init_saver()
        self.padding = 'VALID'
        self.stddev = 0.02
        self.kernel_size = [7,7,3,3,3,3]
        self.conv_num_filters = 256
        self.fc_num_filters = 1024
        self.dropout = FLAGS.dropout
        self.batch_size = FLAGS.batch_size

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.

        pass

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        pass

    def bi_directional_lstm(self,fw_cell,bw_cell,x):


        outputs,_ = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell, x, dtype=tf.float32)


    def character_embedding_layer(self, input, keep_prob, small_or_large="small"):
        # batch_size, max_sents_num, max_sent_size,
        if small_or_large is not "small":
            self.stddev = 0.05
            self.conv_num_filters = 1024
            self.fc_num_filters = 2048

        input = tf.expand_dims(input,-1);
        """h_1 = self._create_conv(input, [self.num_filters],"conv1",3)"""
        h_2 = self._create_conv(h_1, [1, self.kernel_size[1], self.num_filters, self.num_filters],"conv2",3)
        h_3 = self._create_conv(h_2, [1, self.kernel_size[2], self.num_filters, self.num_filters],"conv3")
        h_4 = self._create_conv(h_3, [1, self.kernel_size[3], self.num_filters, self.num_filters],"conv4")
        h_5 = self._create_conv(h_4, [1, self.kernel_size[4], self.num_filters, self.num_filters],"conv5")
        h_6 = self._create_conv(h_5, [1, self.kernel_size[5], self.num_filters, self.num_filters],"conv6",3)



    def _create_conv(self, input, shape, name_scope, pool=None):
        with tf.name_scope(name_scope):

            w = tf.Variable(tf.truncated_normal(shape=shape, stddev=self.stddev, dtype=tf.float32, name='weight'))
            b = tf.Variable(tf.constant(0, shape=shape[-1], dtype=tf.float32, name='bias'))
            conv = tf.nn.conv2d(input=input, filter=w, strides = [1,1,1,1], padding = self.padding)
            activation = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            if pool:
                activtion = tf.nn.max_pool(value=activation,ksize=[1,1,pool,1],strides=[1,1,pool,1],padding=self.padding,name="max_pool")

            return activation

    def _create_fc(self, input, shape, name_scope, keep_prob=None):
        with tf.name_scope(name_scope):

            w = tf.Variable(tf.truncated_normal(shape=shape, stddev=self.stddev, dtype=tf.float32, name='weight'))
            b = tf.Variable(tf.constant(0, shape=shape[-1], dtype=tf.float32, name='bias'))
            fc = tf.nn.bias_add(tf.matmul(input, w), b, name="dense")

            if keep_prob:
                fc = tf.nn.dropout(fc, keep_prob, name="dropout")

            return fc
