from base.base_model import BaseModel
import tensorflow as tf
import Skeleton


tf.flags.DEFINE_integer("batch_size", 128, "minibatch size")
tf.flags.DEFINE_float("dropout", 0.2, "dropout")
tf.flags.DEFINE_integer("character_dim", 70, "chracter dimension size")
tf.flags.DEFINE_integer("glove_word_dim", 300, "word dimension size")
tf.flags.DEFINE_integer("max_sent_num", 60, "max sent numbers in one context")
tf.flags.DEFINE_integer("max_ques_size", 30, "max word numbers in a question")
# tf.flags.DEFINE_integer("max_word_num", 400, "max word numbers in a sentence")
tf.flags.DEFINE_integer("max_word_num", 64, "max word numbers in a sentence")
tf.flags.DEFINE_integer("max_word_size", 16, "max char numbers in a word")
tf.flags.DEFINE_float("keep_prob", 0.8, "keep_prob")

FLAGS = tf.flags.FLAGS

class BiDAFModel(BaseModel, Skeleton):
    def __init__(self, config):
        super(BiDAFModel, self).__init__(config)

        self.build_model()
        self.init_saver()

        self.cnn_config = {
            'filter_size': (7,7,3,3,3,3),
            'conv_num_filters' : 256,
            'fc_num_filters' : 1024,
            'max_sent_size' : 1014,
            'output_dim' : 100
            # 'padding' : 'VALID'
        }

        self.contextual_config = {
            'hidden_state_size': self.cnn_config.output_dim + FLAGS.glove_word_dim,

        }
        self.padding = 'VALID'
        self.stddev = 0.02
        self.dropout = FLAGS.dropout
        self.batch_size = FLAGS.batch_size

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.

        pass

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        pass

    def output_layer():
        eee
    def modeling_layer(self, input,keep_prob):


        JX = tf.shape(input)[2]
        M = tf.shape(input)[1]
        dim = tf.shape(input)[-1]
        lstm_input = tf.reshape(input, [-1,JX,dim]) # [batch * M, JX, 8d] # flatten ----------->
        hidden_size = self.contextual_config.hidden_state_size
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size , state_is_tuple=True)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)

        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)
        """ first layer
        """
        outputs_1,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, lstm_input, dtype = tf.float32)

        outputs_fw_1 = outputs_1[0]
        outputs_bw_1 = outputs_1[1]

        outputs_fw_1 = tf.reshape(outputs_fw_1,[-1,M,JX,hidden_size]) # restore <--------
        outputs_bw_1 = tf.reshape(outputs_bw_1,[-1,M,JX,hidden_size]) # restore <--------
        outputs_2d_1 = tf.concat([outputs_fw_1,outputs_bw_1],axis=-1) # [batch, M, JX, 2d] # d = 100
        """ second layer
        """
        outputs_2,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, outputs_2d_1, dtype = tf.float32)

        outputs_fw_2 = outputs_2[0]
        outputs_bw_2 = outputs_2[1]

        outputs_fw_2 = tf.reshape(outputs_fw_2,[-1,M,JX,hidden_size]) # restore <--------
        outputs_bw_2 = tf.reshape(outputs_bw_2,[-1,M,JX,hidden_size]) # restore <--------
        outputs_2d_2 = tf.concat([outputs_fw_2,outputs_bw_2],axis=-1) # [batch, M, JX, 2d] # d = 100
        return outputs_2d_2
    def attention_flow_layer(self, input_q, input_c, keep_prob):

        # input_c : [batch, M, JX, 2d]
        # input_q : [batch, JQ, 2d]

        JX = tf.shape(input_c)[2]
        M = tf.shape(input_c)[1]
        JQ = tf.shape(input_q)[1]
        batch = tf.shape(input_c)[0]
        input_c_aug = tf.tile(tf.expand_dims(input_c, 3), [1, 1, 1, JQ, 1]) # [batch, M, JX, JQ, 2d]
        input_q_aug = tf.tile(tf.expand_dims(tf.expand_dims(input_q, 1), 1), [1, M, JX, 1, 1])

        d_6 = tf.concat([input_c_aug,input_q_aug,input_c_aug*input_q_aug], axis = -1) # [batch, M, JX, JQ, 6d]
        d_6 = tf.nn.dropout(d_6, keep_prob)
        w = tf.Variable(tf.truncated_normal(shape=d_6.get_shape()[-1], stddev=self.stddev, dtype=tf.float32, name='weight'))
        w_aug = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(w,-1),0),0),0), [batch,M,JX,1,1])
        # [batch, M, JX, 6d, 1]

        s_tj = tf.matmul(d_6,w_aug) # [batch, M, JX, JQ, 1]
        org_shape = s_tj.get_shape()
        flatten_s_tj = tf.reshape(s_tj, [-1,s_tj.get_shape()[-1]])
        softmaxed_s_tj = tf.reshape(tf.nn.softmax(flatten_s_tj), org_shape)
        u_t = tf.reduce_sum(softmaxed_s_tj*input_q_aug,3) # [batch, M, JX, 2d]

        s_tj = tf.reshape(s_tj, [batch,M,JX,JQ])
        maxpooled_s_tj = tf.nn.max_pool(s_tj,ksize=[1,1,1,JQ],strides=[1,1,1,JQ], padding='SAME')
        flatten_s_tj = tf.reshape(maxpooled_s_tj,[-1, maxpooled_s_tj.get_shape()[-2]])
        b_t = tf.reshape(tf.nn.softmax(flatten_s_tj), [batch, M, JX, 1]) # [batch, M, JX, 1]
        # tf.tile(tf.expand_dims(b_t, 3),[1,1,1,JQ]
        h_bar = b_t*input_c # [batch, M,JX, 2d]

        g_t = tf.concat([input_c, u_t, input_c*u_t, input_c*h_bar],axis = -1) # [batch, M, JX, 8d]
        return g_t



    def contextual_embedding_layer(self, embedded_character, embedded_word, is_context=True,keep_prob):

        lstm_input = tf.concat([embedded_word, embedded_character], axis = -1) # [batch, msn, mwn, w_d] , [batch, msn, mwn, c_d]
        # previous_d = w_d + c_d = 300 + 100 = 400
        prev_d = embedded_word.get_shape()[-1] + embedded_character.get_shape()[-1]
        # hidden_size = 100
        # lstm cell 생성
        if is_context:
            lstm_input = tf.reshape(lstm_input, [-1,FLAGS.max_word_num,prev_d]) # flatten ------------->

        hidden_size = self.contextual_config.hidden_state_size
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size , state_is_tuple=True)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)

        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)

        outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, lstm_input, dtype = tf.float32)

        outputs_fw = outputs[0]
        outputs_bw = outputs[1]

        if is_context:
            outputs_fw = tf.reshape(outputs_fw,[-1,max_sent_num,FLAGS.max_word_num,hidden_size]) # restore <--------
            outputs_bw = tf.reshape(outputs_bw,[-1,max_sent_num,FLAGS.max_word_num,hidden_size]) # restore <--------
            outputs_2d = tf.concat([outputs_fw,outputs_bw],axis=-1) # [batch, msn, mwn, d] # d = 100
        else:
            outputs_2d = tf.concat([outputs_fw,outputs_bw],axis=-1) # [batch, mwn, d]

        return outputs_2d

    def character_embedding_layer(self, input_x, keep_prob, small_or_large="small"):
        # batch_size, max_sent_num, max_sent_size,
        # input_x is a word.
        config = self.cnn_conofig

        if small_or_large is not "small":
            config.stddev = 0.05
            config.conv_num_filters = 1024
            config.fc_num_filters = 2048

        # input_x = tf.placeholder(tf.float32, [None, 70, max_sent_size, 1], name="input_x")
        # input_x = tf.expand_dims(input,-1);
        h_1 = self._create_conv(input_x, [Flags.character_dim, config.filter_size[0], 1, config.conv_num_filters],"conv1-max-pooled",3)
        h_2 = self._create_conv(h_1, [1, config.filter_size[1], config.conv_num_filters, config.conv_num_filters],"conv2-max-pooled",3)
        h_3 = self._create_conv(h_2, [1, config.filter_size[2], config.conv_num_filters, config.conv_num_filters],"conv3")
        h_4 = self._create_conv(h_3, [1, config.filter_size[3], config.conv_num_filters, config.conv_num_filters],"conv4")
        h_5 = self._create_conv(h_4, [1, config.filter_size[4], config.conv_num_filters, config.conv_num_filters],"conv5")
        h_6 = self._create_conv(h_5, [1, config.filter_size[5], config.conv_num_filters, config.conv_num_filters],"conv6-max-pooled",3)

        print(h_6.get_shape())
        total_features = h_6.get_shape()[2] * config.conv_num_filters
        h_flat = tf.reshape(h_6, [-1, total_features])

        h_7 = self._create_fc(h_flat, [total_features, config.fc_num_filters], "fc1", keep_prob)
        h_8 = self._create_fc(h_7, [config.fc_num_filters, config.fc_num_filters], "fc2", keep_prob)
        h_9 = self._create_fc(h_8, [config.fc_num_filters, config.output_dim], "fc3")

        return h_9

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
