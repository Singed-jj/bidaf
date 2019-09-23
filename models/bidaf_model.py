from base.base_model import BaseModel
import tensorflow as tf
# import Skeleton


# tf.flags.DEFINE_integer("batch_size", , "minibatch size")
tf.flags.DEFINE_float("dropout", 0.2, "dropout")
tf.flags.DEFINE_integer("char_dim", 100, "chracter dimension size")
tf.flags.DEFINE_integer("glove_word_dim", 300, "word dimension size")
tf.flags.DEFINE_integer("max_sent_num", 60, "max sent numbers in one context")
tf.flags.DEFINE_integer("max_ques_size", 30, "max word numbers in a question")
# tf.flags.DEFINE_integer("max_word_num", 400, "max word numbers in a sentence")
tf.flags.DEFINE_integer("max_word_num", 64, "max word numbers in a sentence")
tf.flags.DEFINE_integer("max_word_size", 16, "max char numbers in a word")
tf.flags.DEFINE_float("keep_prob", 0.8, "keep_prob")
tf.flags.DEFINE_integer("total_char_num", 70, "there is 70 char")
tf.flags.DEFINE_integer("total_word_num", 2200000, "<- number of words embedded")
tf.flags.DEFINE_string("mode","","")
FLAGS = tf.flags.FLAGS
#FLAGS.mode = "train"

def get_model(config):
    with tf.name_scope("model_cpu0") as scope, tf.device("/cpu:0"):
        model = BiDAFModel(config, scope)
        tf.get_variable_scope().reuse_variables()
            # models.append(model)
    return model

class BiDAFModel(BaseModel):
    def __init__(self, config, scope):
        super(BiDAFModel, self).__init__(config)

        self.scope = scope
        self.config = config
        self.global_step = tf.compat.v1.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        N, M, JX, JQ, CD, W = \
            self.config.batch_size, FLAGS.max_sent_num, FLAGS.max_word_num, \
            FLAGS.max_ques_size, FLAGS.char_dim, FLAGS.max_word_size


        self.cnn_config = {
            'filter_size': (7,7,3,3,3,3),
            'conv_num_filters' : 256,
            'fc_num_filters' : 1024,
            'max_sent_size' : 1014,
            'output_dim' : 100
            # 'padding' : 'VALID'
        }

        self.contextual_config = {
            'hidden_state_size': 100,

        }
        self.padding = 'VALID'
        self.stddev = 0.02
        self.dropout = FLAGS.dropout
        self.x_w = tf.placeholder('int32', [N, None, None], name='xw')
        self.x_c = tf.placeholder('int32', [N, None, None, W], name='xc')
        self.x_qw = tf.placeholder('int32', [N, None], name='qw')
        self.x_qc = tf.placeholder('int32', [N, None, W], name='qc')
        self.y_1 = tf.placeholder('bool', [N, None, None], name='y1')
        self.y_2 = tf.placeholder('bool', [N, None, None], name='y2')
        self.dropout = tf.placeholder('float32', name='dropout_keep_prob')



        self.build_model(self.dropout)
        self.init_saver()

    def build_model(self, keep_prob):
        # here you build the tensorflow graph of any model you want and also define the loss.
        N, M, JX, JQ, WN, CN, CD, W = \
            self.config.batch_size, FLAGS.max_sent_num, FLAGS.max_word_num, FLAGS.max_ques_size,\
            FLAGS.total_word_num, FLAGS.total_char_num, FLAGS.char_dim, FLAGS.max_word_size

        with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
            char_emb_mat = tf.compat.v1.get_variable("char_emb_mat", shape=[CN, CD], dtype='float')

        with tf.variable_scope("char"):
            xc = tf.nn.embedding_lookup(char_emb_mat, self.x_c)  # [N, M, JX, W, 100]
            xqc = tf.nn.embedding_lookup(char_emb_mat, self.x_qc)  # [N, JQ, W, 100]
            xc = tf.reshape(xc, [-1, M*JX, W, CD])
            xqc = tf.reshape(xqc, [-1, JQ, W, CD])

            tf.compat.v1.get_variable_scope().reuse_variables()
            xc_cnned = self.character_embedding_layer(xc, keep_prob) # [N, M, JX, d]
            xqc_cnned = self.character_embedding_layer(xqc, keep_prob,is_context=False)

        with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
            if self.config.mode == 'train':
                word_emb_mat = tf.compat.v1.get_variable("word_emb_mat", dtype='float', initializer=self.config.emb_mat)
            else:
                word_emb_mat = tf.compat.v1.get_variable("word_emb_mat", dtype='float')
        with tf.name_scope("word"):

            xw = tf.nn.embedding_lookup(word_emb_mat, self.x_w)  # [N, M, JX, 300]
            qw = tf.nn.embedding_lookup(word_emb_mat, self.x_qw)  # [N, JQ, 300]


        xc_contexed = self.contextual_embedding_layer(xc_cnned, xw, keep_prob,"context_embed_0")
        xq_contexed = self.contextual_embedding_layer(xqc_cnned, qw, keep_prob,"context_embed_1",is_context=False)

        x_g = self.attention_flow_layer(xq_contexed, xc_contexed, keep_prob)
        x_m1, x_m2 = self.modeling_layer(x_g,keep_prob)

        self.output_layer(x_m1, x_m2, x_g)


    def build_loss(self):
        loss_p1 = tf.nn.softmax_cross_entropy_with_logits(
            self.logit_p1, tf.cast(tf.reshape(self.y_1, [-1, M * JX]), 'float'))
        loss_p2 = tf.nn.softmax_cross_entropy_with_logits(
            self.logit_p2, tf.cast(tf.reshape(self.y_2, [-1, M * JX]), 'float'))

        self.loss = tf.reduce_mean(loss_p1) + tf.reduce_mean(loss_p2)

    def get_initializer(self,matrix):
        def _initializer(shape, dtype=None, partition_info=None, **kwargs): return matrix
        return _initializer
        # self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
        # tf.scalar_summary(self.loss.op.name, self.loss)
    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        pass

    def output_layer(self, input_m1, input_m2, input_g):

        with tf.name_scope("output_layer"):
            batch, M, JX, _ = tf.shape(input_m1)

            dim = tf.shape(input_m1)[-1] + tf.shape(input_g)[-1]

            # input = Modeling layer output # [batch, M, JX, 2d] # d = 100
            # shape = dim
            w_1 = tf.Variable(tf.truncated_normal(shape=[dim,1], stddev=self.stddev, dtype=tf.float32, name='weight_1'))
            w_1 = tf.tile(tf.expand_dims(w_1,0),[batch,1,1])
            concat_m1_g = tf.reshape(tf.concat([input_m1, input_g], axis = -1),[-1, M*JX,dim])

            w_2 = tf.Variable(tf.truncated_normal(shape=[dim,1], stddev=self.stddev, dtype=tf.float32, name='weight_2'))
            w_2 = tf.tile(tf.expand_dims(w_2,0),[batch,1,1])
            concat_m2_g = tf.reshape(tf.concat([input_m2, input_g], axis = -1),[-1, M*JX,dim])

            p1 = softmax(tf.reshpae(tf.matmul(concat_m1_g,w)), [-1, M*JX]) # [batch, M*JX]
            p2 = softmax(tf.reshape(tf.matmul(concat_m2_g,w)), [-1, M*JX])

            self.logit_p1 = p1
            self.logit_p2 = p2

            self.y_p1 = tf.reshape(p1, [-1,M,JX])
            self.y_p2 = tf.reshape(p2, [-1,M,JX])



    def modeling_layer(self, input_g,keep_prob):

        with tf.name_scope("modeling_layer"):

            JX = tf.shape(input_g)[2]
            M = tf.shape(input_g)[1]
            dim = tf.shape(input_g)[-1]
            lstm_input = tf.reshape(input_g, [-1,JX,dim]) # [batch * M, JX, 8d] # flatten ----------->
            hidden_size = self.contextual_config['hidden_state_size']
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size , state_is_tuple=True)
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)

            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)
            """ first layer
            """
            print(f'first: {lstm_input.get_shape()}')
            outputs_1,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, lstm_input, dtype = tf.float32)

            outputs_fw_1 = outputs_1[0]
            outputs_bw_1 = outputs_1[1]

            outputs_fw_1 = tf.reshape(outputs_fw_1,[-1,JX,hidden_size]) # restore <--------
            outputs_bw_1 = tf.reshape(outputs_bw_1,[-1,JX,hidden_size]) # restore <--------
            outputs_2d_1 = tf.concat([outputs_fw_1,outputs_bw_1],axis=-1) # [batch * M, JX, 2d] # d = 100

            """ second layer(M_1)
            """
            print(f'second: {outputs_2d_1.get_shape()}')
            outputs_2,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, outputs_2d_1, dtype = tf.float32)

            outputs_fw_2 = outputs_2[0]
            outputs_bw_2 = outputs_2[1]

            outputs_fw_2 = tf.reshape(outputs_fw_2,[-1,JX,hidden_size]) # restore <--------
            outputs_bw_2 = tf.reshape(outputs_bw_2,[-1,JX,hidden_size]) # restore <--------
            outputs_m1 = tf.concat([outputs_fw_2,outputs_bw_2],axis=-1) # [batch * M, JX, 2d] # d = 100

            """ another layer(M_2)
            """
            print(f'third: {outputs_m1.get_shape()}')
            outputs_3,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, outputs_m1, dtype = tf.float32)

            outputs_fw_3 = outputs_3[0]
            outputs_bw_3 = outputs_3[1]

            outputs_fw_3 = tf.reshape(outputs_fw_3,[-1,M,JX,hidden_size]) # restore <--------
            outputs_bw_3 = tf.reshape(outputs_bw_3,[-1,M,JX,hidden_size]) # restore <--------
            outputs_m2 = tf.concat([outputs_fw_3,outputs_bw_3],axis=-1) # [batch, M, JX, 2d] # d = 100
            return tf.reshape(outputs_m1,[-1,M,JX,2*hidden_size]), outputs_m2

    def attention_flow_layer(self, input_q, input_c, keep_prob):

        # input_c : [batch, M, JX, 2d]
        # input_q : [batch, JQ, 2d]
        with tf.name_scope("attention_flow_layer"):
            JX = tf.shape(input_c)[2]
            M = tf.shape(input_c)[1]
            JQ = tf.shape(input_q)[1]
            batch = tf.shape(input_c)[0]
            input_c_aug = tf.tile(tf.expand_dims(input_c, 3), [1, 1, 1, JQ, 1]) # [batch, M, JX, JQ, 2d]
            input_q_aug = tf.tile(tf.expand_dims(tf.expand_dims(input_q, 1), 1), [1, M, JX, 1, 1])

            print(f'ic: {input_c.get_shape()}')
            print(f'iq: {input_q.get_shape()}')
            print(f'ica: {input_c_aug.get_shape()}')
            print(f'iqa: {input_q_aug.get_shape()}')
            d_6 = tf.concat([input_c_aug,input_q_aug,input_c_aug*input_q_aug], axis = -1) # [batch, M, JX, JQ, 6d]
            d_6 = tf.nn.dropout(d_6, keep_prob)

            w = tf.Variable(tf.truncated_normal(shape=[tf.shape(d_6)[-1]], \
                stddev=self.stddev, dtype=tf.float32, name='weight'))
            w_aug = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(w,-1),0),0),0), [batch,M,JX,1,1])
            # [batch, M, JX, 6d, 1]

            s_tj = tf.matmul(d_6,w_aug) # [batch, M, JX, JQ, 1]
            org_shape = s_tj.get_shape()
            flatten_s_tj = tf.reshape(s_tj, [-1,s_tj.get_shape()[-1]])
            softmaxed_s_tj = tf.reshape(tf.nn.softmax(flatten_s_tj), org_shape)
            u_t = tf.reduce_sum(softmaxed_s_tj*input_q_aug,3) # [batch, M, JX, 2d]

            s_tj = tf.reshape(s_tj, [batch,M,JX,JQ])

            maxpooled_s_tj = tf.nn.max_pool2d(s_tj,ksize=[1,1,1,FLAGS.max_ques_size],strides=[1,1,1,FLAGS.max_ques_size], padding='SAME')
            flatten_s_tj = tf.reshape(maxpooled_s_tj,[-1, maxpooled_s_tj.get_shape()[-2]])
            b_t = tf.reshape(tf.nn.softmax(flatten_s_tj), [batch, M, JX, 1]) # [batch, M, JX, 1]
            # tf.tile(tf.expand_dims(b_t, 3),[1,1,1,JQ]
            h_bar = b_t*input_c # [batch, M,JX, 2d]

            g_t = tf.concat([input_c, u_t, input_c*u_t, input_c*h_bar],axis = -1) # [batch, M, JX, 8d]
            return g_t



    def contextual_embedding_layer(self, embedded_character, embedded_word, keep_prob,scope,reuse=False,is_context=True):

        with tf.name_scope(scope), tf.variable_scope(scope):
            # if reuse:
            #     tf.get_variable_scope().reuse_variables()
            # print(f'ew: {embedded_word.get_shape()}')
            # print(f'ec: {embedded_character.get_shape()}')
            lstm_input = tf.concat([embedded_word, embedded_character], axis = -1) # [batch, msn, mwn, w_d] , [batch, msn, mwn, c_d]
            # previous_d = w_d + c_d = 300 + 100 = 400
            prev_d = embedded_word.get_shape()[-1] + embedded_character.get_shape()[-1]
            # hidden_size = 100
            # lstm cell 생성
            if is_context:
                lstm_input = tf.reshape(lstm_input, [-1,FLAGS.max_word_num,prev_d]) # flatten ------------->

            hidden_size = self.contextual_config['hidden_state_size']
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size , state_is_tuple=True)
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)

            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)

            outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, lstm_input, dtype = tf.float32)

            outputs_fw = outputs[0]
            outputs_bw = outputs[1]

            if is_context:
                outputs_fw = tf.reshape(outputs_fw,[-1,FLAGS.max_sent_num,FLAGS.max_word_num,hidden_size]) # restore <--------
                outputs_bw = tf.reshape(outputs_bw,[-1,FLAGS.max_sent_num,FLAGS.max_word_num,hidden_size]) # restore <--------
                outputs_2d = tf.concat([outputs_fw,outputs_bw],axis=-1) # [batch, msn, mwn, d] # d = 100
            else:
                outputs_2d = tf.concat([outputs_fw,outputs_bw],axis=-1) # [batch, mwn, d]

            return outputs_2d

    def character_embedding_layer(self, input_x, keep_prob,is_context=True, small_or_large="small"):
        # batch_size, max_sent_num, max_sent_size,
        with tf.name_scope("character_embedding_layer"):
            config = self.cnn_config

            if small_or_large is not "small":
                config['stddev'] = 0.05
                config['conv_num_filters'] = 1024
                config['fc_num_filters'] = 2048


            input_x = tf.transpose(input_x, perm=[0,3,2,1])
            if is_context:
                input_x = tf.reshape(input_x,shape=[-1,FLAGS.char_dim,FLAGS.max_sent_num*FLAGS.max_word_num*FLAGS.max_word_size,1])
            else:
                input_x = tf.reshape(input_x,shape=[-1,FLAGS.char_dim,FLAGS.max_ques_size*FLAGS.max_word_size,1])
            # input_x = tf.placeholder(tf.float32, [None, 70, max_sent_size, 1], name="input_x")
            # input_x = tf.expand_dims(input,-1);
            h_1 = self._create_conv(input_x, \
                [FLAGS.char_dim, config['filter_size'][0], 1, config['conv_num_filters']],"conv1-max-pooled",3)
            h_2 = self._create_conv(h_1, \
                [1, config['filter_size'][1], config['conv_num_filters'], config['conv_num_filters']],"conv2-max-pooled",3)
            h_3 = self._create_conv(h_2, \
                [1, config['filter_size'][2], config['conv_num_filters'], config['conv_num_filters']],"conv3")
            h_4 = self._create_conv(h_3, \
                [1, config['filter_size'][3], config['conv_num_filters'], config['conv_num_filters']],"conv4")
            h_5 = self._create_conv(h_4, \
                [1, config['filter_size'][4], config['conv_num_filters'], config['conv_num_filters']],"conv5")
            h_6 = self._create_conv(h_5, \
                [1, config['filter_size'][5], config['conv_num_filters'], config['conv_num_filters']],"conv6-max-pooled",3)

            total_features = h_6.get_shape()[2] * config['conv_num_filters']
            h_flat = tf.reshape(h_6, [-1, total_features])

            h_7 = self._create_fc(h_flat, [tf.cast(total_features,'int32'), config['fc_num_filters']], "fc1", keep_prob)
            h_8 = self._create_fc(h_7, [config['fc_num_filters'], config['fc_num_filters']], "fc2", keep_prob)
            h_9 = self._create_fc(h_8, [config['fc_num_filters'], config['output_dim']], "fc3")

            if is_context:
                h_9 = tf.reshape(h_9, [-1,FLAGS.max_sent_num,FLAGS.max_word_num, config['output_dim']])
            else:
                h_9 = tf.reshape(h_9, [-1,FLAGS.max_ques_size, config['output_dim']])

            return h_9

    def _create_conv(self, input, shape, name_scope, pool=None):
        with tf.name_scope(name_scope):
            
            w = tf.Variable(tf.random.truncated_normal(shape=shape, stddev=self.stddev, dtype=tf.float32, name='weight'))
            b = tf.Variable(tf.constant(0, shape=[shape[-1]], dtype=tf.float32, name='bias'))
            conv = tf.nn.conv2d(input=input, filter=w, strides = [1,1,1,1], padding = self.padding)
            activation = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            if pool:
                activtion = tf.nn.max_pool2d(activation, \
                    ksize=[1,1,pool,1],strides=[1,1,pool,1],padding=self.padding,name="max_pool")

            return activation

    def _create_fc(self, input, shape, name_scope, keep_prob=None):
        with tf.name_scope(name_scope):

            w = tf.Variable(tf.truncated_normal(shape=shape, stddev=self.stddev, dtype=tf.float32, name='weight'))
            b = tf.Variable(tf.constant(0, shape=[shape[-1]], dtype=tf.float32, name='bias'))
            fc = tf.nn.bias_add(tf.matmul(input, w), b, name="dense")

            if keep_prob!= None:
                fc = tf.nn.dropout(fc, keep_prob, name="dropout")

            return fc


    def feed_x_and_y(self, pre_feed_dict, mode=None): # (typeof batch == json) # True

        feed_dict = {}
        feed_dict[self.x_w] = pre_feed_dict['x_w']
        feed_dict[self.x_c] = pre_feed_dict['x_c']
        feed_dict[self.x_qw] = pre_feed_dict['x_qw']
        feed_dict[self.x_qc] = pre_feed_dict['x_qc']
        feed_dict[self.y_1] = pre_feed_dict['y_1']
        feed_dict[self.y_2] = pre_feed_dict['y_2']
        feed_dict[self.dropout] = pre_feed_dict['dropout']
        return feed_dict

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
