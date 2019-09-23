import argparse
import json
import math
import os
import shutil
from pprint import pprint
from collections import namedtuple
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import sys
from trainers.bidaf_trainer import BiDAFTrainer
from models.bidaf_model import BiDAFModel
from models.bidaf_model import get_model
from utils.logger import Logger


sys.path.append("./")
def main(config):
    """
    config 에 디렉토리 추가해주고
    """

    if not os.path.exists(config.summary_dir):
        os.makedirs(config.summary_dir)

    if config.mode == 'train':
        _train(config)
    elif config.mode == 'test':
        _test(config)
    else:
        raise ValueError("invalid value for 'mode': {}".format(config.mode))


def _train(config):

    total_path = os.path.join(config.data_dir, "fixed_data_total.json")

    train_data_path = os.path.join(config.data_dir, "data_train.json")
    train_fixed_data_path = os.path.join(config.data_dir, "fixed_data_train.json")
    with open(train_data_path, 'r') as fh:
        train_data = json.load(fh)
    with open(train_fixed_data_path, 'r') as fh:
        train_fixed_data = json.load(fh)

    dev_data_path = os.path.join(config.data_dir, "data_dev.json")
    dev_fixed_data_path = os.path.join(config.data_dir, "fixed_data_dev.json")
    with open(dev_data_path, 'r') as fh:
        dev_data = json.load(fh)
    with open(dev_fixed_data_path, 'r') as fh:
        dev_fixed_data = json.load(fh)


    # train_data = np.load(train_data_path)
    # dev_data = np.load(dev_data_path)

    '''degug'''
    # test_data_path = os.path.join(config.data_dir, "data_test.json")
    # test_fixed_data_path = os.path.join(config.data_dir, "fixed_data_test.json")
    # with open(test_data_path, 'r') as fh:
    #     test_data = json.load(fh)
    # with open(test_fixed_data_path, 'r') as fh:
    #     test_fixed_data = json.load(fh)
    #
    # train_data = test_data
    # train_fixed_data = test_fixed_data
    ''''''

    total_data = json.load(open(total_path,'r'))
    total_w2v_dict = total_data['w2v_dict']
    total_w2i_dict = total_data['w2i_dict']
    total_c2i_dict = total_data['c2i_dict']

    assert len(total_w2v_dict) == len(total_w2i_dict)
    i2v_dict = {total_w2i_dict[word]: vec for i, (word, vec) in enumerate(total_w2v_dict.items()) if word in total_w2i_dict}


    emb_mat = np.array([i2v_dict[i] if i in i2v_dict \
                                    else np.zeros(config.word_emb_size) \
                                    for i in tqdm(range(len(total_w2v_dict)))
                                    ])



    config.emb_mat = tf.convert_to_tensor(emb_mat,dtype='float')
    print(f"embedding done : {config.emb_mat.get_shape()}")
    x_i = train_data['x_i']
    qs_w = tf.ragged.constant(train_data['qs_w'])
    qs_c = tf.ragged.constant(train_data['qs_c'])
    y = train_data['y']
    x_aw = tf.ragged.constant(train_fixed_data['x_aw'])
    x_ac = tf.ragged.constant(train_fixed_data['x_ac'])

    train_context = {
        'w2i_dict': total_w2i_dict,
        'c2i_dict': total_c2i_dict,
        'x_aw': x_aw,
        'x_ac': x_ac
    }

    dataset = tf.data.Dataset.from_tensor_slices({
        'x_i':x_i,
        'qs_w':qs_w,
        'qs_c':qs_c,
        'y':y})
    dataset = dataset.shuffle(1000000).repeat().batch(config.batch_size)

    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    dev_context = {
        'w2i_dict': total_w2i_dict,
        'c2i_dict': total_c2i_dict,
        'x_aw':tf.ragged.constant(dev_fixed_data['x_aw']),
        'x_ac':tf.ragged.constant(dev_fixed_data['x_ac'])
    }
    dev_batch = {
        'x_i':dev_data['x_i'],
        'qs_w':tf.ragged.constant(dev_data['qs_w']),
        'qs_c':tf.ragged.constant(dev_data['qs_c']),
        'y':dev_data['y']}
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
    model = get_model(config)
    logger = Logger(sess,config)
    trainer = BiDAFTrainer(model,config,logger)

    num_steps = int(math.ceil(len(y) / config.batch_size)) * config.num_epochs
    for i in tqdm(range(num_steps),mininterval=1):
    # for _ in tqdm( total = num_stps):
        batch = sess.run(next_batch)
        global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
        get_summary = global_step % config.log_period == 0
        loss, summary, train_op = trainer.step(sess, batch, context, get_summary=get_summary)

        if global_step % config.dev_period == 0:
            print("\nEvaluation:")
            trainer.dev_step(sess, dev_batch, context, get_summary=get_summary)
            print("")



def _test(config):
    pass

class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

def _run():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="config.json path")
    parser.add_argument("--mode",required=True,help="train? test?")

    args = parser.parse_args()
    print(f'args : {args}')
    with open(args.config_path, 'r') as fh:
        config = Config(**json.load(fh))
        config.mode = args.mode
        main(config)

if __name__ == '__main__':
    _run()
