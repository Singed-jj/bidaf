import nltk
sent_tokenize = nltk.sent_tokenize
import tensorflow as tf
from functools import reduce
from operator import mul
import numpy as np

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

# context = "The state also has five Micropolitan Statistical Areas centered on Bozeman, Butte, Helena, Kalispell and Havre. These communities, excluding Havre, are colloquially known as the \"big 7\" Montana cities, as they are consistently the seven largest communities in Montana, with a significant population difference when these communities are compared to those that are 8th and lower on the list. According to the 2010 U.S. Census, the population of Montana's seven most populous cities, in rank order, are Billings, Missoula, Great Falls, Bozeman, Butte, Helena and Kalispell. Based on 2013 census numbers, they collectively contain 35 percent of Montana's population. and the counties containing these communities hold 62 percent of the state's population. The geographic center of population of Montana is located in sparsely populated Meagher County, in the town of White Sulphur Springs."
# xi = list(map(word_tokenize, sent_tokenize(context)))
#
# cxi = [[list(xijk) for xijk in xij] for xij in xi]
# # print(xi)
# # print("==========")
# # print(cxi)
# qi = word_tokenize("this is an example statement")
# print(qi)

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out

def get_2d_spans(text, tokenss):
    spanss = []
    cur_idx = 0
    for tokens in tokenss:
        spans = []
        for token in tokens:
            if text.find(token, cur_idx) < 0:
                print(tokens)
                print("{} {} {}".format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        spanss.append(spans)
    return spanss



# spanss = get_2d_spans(context,xi)
# idxs = []
#
# start = 381
# stop = 464
# for sent_idx, spans in enumerate(spanss):
#     for word_idx, span in enumerate(spans):
#         if not (stop <= span[0] or start >= span[1]):
#             idxs.append((sent_idx, word_idx))
# assert len(idxs) > 0, "{} {} {} {}".format(context, spanss, start, stop)
# res = idxs[0], (idxs[-1][0], idxs[-1][1] + 1)
# print("spanss")
# print(spanss)
# print("res : ",res)

def see_output_shape():
    hidden_size = 100
    batch_size = 64
    max_sent_size = 1014
    d = 300 + 100
    sents_in_context = 120
    input = tf.Variable(tf.constant(1.,shape = [batch_size,sents_in_context,max_sent_size,d]))

    # lstm_input = flatten(input,2)
    lstm_input = input
    print("lstm_input :", lstm_input.get_shape())
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size , state_is_tuple=True)
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell)

    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell)
    outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, lstm_input, dtype = tf.float32)

    # fw = reconstruct(outputs[0], input,2)
    # bw = reconstruct(outputs[1], input,2)

    print("forward_shape : ",fw.get_shape())



def main():

    see_output_shape()



if __name__ == "__main__":
    main()
