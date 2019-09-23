from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import time
import datetime


class BiDAFTrainer(BaseTrain):
    def __init__(self, model, data, config, logger):
        super(BiDAFTrainer, self).__init__(sess, model, config, logger)

        # self.sess = sess
        self.model = model
        # self.data = data
        self.config = config
        self.summamry = True
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.5)
        self.loss = model.get_loss()
        self.global_step = model.get_global_step()
        self.grads = self.opt.compute_gradients(self.loss)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop on the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        pass

    def train_step(self, sess, batch, context, summary =False):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        N, M, JX, JQ, CD, W = \
            self.config.batch_size, self.config.max_sent_num, self.config.max_word_num, \
            self.config.max_ques_size, self.config.character_dim, self.config.max_word_size

        # batch['data']['x_i']
        pre_feed_dict = self.lookup(batch, context)
        pre_feed_dict['dropout'] = 0.2
        feed_dict = self.model.feed_dict(pre_feed_dict, mode="train")

        if summary:
            loss, summary, train_op = \
                    sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None

        return loss, summary, train_op


    def dev_step(self, sess, batch, context, summary=False):

        N, M, JX, JQ, CD, W = \
            self.config.batch_size, self.config.max_sent_num, self.config.max_word_num, \
            self.config.max_ques_size, self.config.character_dim, self.config.max_word_size

        # batch['data']['x_i']
        pre_feed_dict = self.lookup(batch, context)
        pre_feed_dict['dropout'] = 1.0
        feed_dict = self.model.feed_dict(pre_feed_dict, mode="dev")

        if summary:
            step, loss, summary = sess.run([self.global_step, self.loss, self.summary], feed_dict=feed_dict)
        else:
            step, loss = sess.run([self.global_step, self.loss], feed_dict=feed_dict)
            summary = None

        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss))

        return loss, summary


    def lookup(batch, context):

        N, M, JX, JQ, CD, W = \
            self.config['batch_size'], self.config['max_sent_num'], self.config['max_word_num'], \
            self.config['max_ques_size'], self.config['character_dim'], self.config['max_word_size']

        art_parag_tuple = batch['x_i']

        w2i_dict = context['w2i_dict']
        c2i_dict = context['c2i_dict']

        x_w = np.full([N, M, JX], -1,dtype=int32)
        x_c = np.empty([N, M, JX, W], -1,dtype=int32)

        x_qw = np.empty([N, JQ], -1,dtype=int32)
        x_qc = np.empty([N, JQ, W], -1,dtype=int32)

        y_1 = np.empty([N, M, JX], dtype='bool')
        y_2 = np.empty([N, M, JX], dtype='bool')
        for n, (art_num, parag_num) in enumerate(tqdm(art_parag_tuple,total=N)):
            tmp_w = context['x_aw'][art_num][parag_num] # [sent_num, word_num]
            tmp_c = context['x_ac'][art_num][parag_num] # [sent_num, word_num, char_num]

            for m in range(M):
                if m >= len(tmp_w): break
                for jx in range(JX):
                    if jx >= len(tmp_w[m]): break
                    if tmp_w[m][jx] in w2i_dict:
                        x_w[n,m,jx] = w2i_dict[tmp_w[m][jx]]
                    for w in range(W):
                        if w >= len(tmp_c[m][jx]): break
                        if tmp_c[m][jx][w] in c2i_dict:
                            x_c[n,m,jx,w] = c2i_dict[tmp_c[m][jx][w]]
        for n in range(N):
            tmp_qw = batch['qs_w'][n]
            tmp_qc = batch['qs_c'][n]
            for jq in range(JQ):
                if jq >= len(tmp_qw): break
                if tmp_qw[jq] in w2i_dict:
                    x_qw[n,jq] = w2i_dict[tmp_qw[jq]]
                for w in range(W):
                    if w >= len(tmp_qc[jq]): break
                    if tmp_qc[jq][w] in c2i_dict:
                        x_qc[n,jq,w] = c2i_dict[tmp_qc[jq][w]]

        cnt = 0
        for n in range(N):
            tmp_y1, tmp_y2 = batch['y'][n][0]
            if tmp_y1[0] >= M or tmp_y1[1] >= JX or tmp_y2[0] >= M or tmp_y2[1] >= JX:
                y_1[n,M-1,JX-1] = True
                y_2[n,M-1,JX-1] = True
                cnt +=1
                continue
            y_1[n,tmp_y1[0],tmp_y1[1]] = True
            y_2[n,tmp_y2[0],tmp_y2[1]] = True

        pre_feed_dict = {}
        pre_feed_dict['x_w'] = x_w
        pre_feed_dict['x_c'] = x_c
        pre_feed_dict['x_qw'] = x_qw
        pre_feed_dict['x_qc'] = x_qc
        pre_feed_dict['y_1'] = y_1 # clear
        pre_feed_dict['y_2'] = y_2 # clear

        return pre_feed_dict
