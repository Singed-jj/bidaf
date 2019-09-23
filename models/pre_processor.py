import os
import json
import nltk
from tqdm import tqdm
import numpy as np

args = {
    'glove_corpus' : '840B',
    'glove_vec_dim' : 300,
    'glove_dir' : '../data/glove/',
    'squad_dir' : '../data/squad/',
    'train_ratio' : 0.9
}

sent_tokenize = nltk.sent_tokenize
def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

def get_w2v_dict(word_set):
    glove_path = os.path.join(args['glove_dir'], "glove.{}.{}d.txt".format(args['glove_corpus'], args['glove_vec_dim']))
    num_word = 2200000
    w2v = dict()
    w2i = dict()
    # num_word = 100
    with open(glove_path, 'r') as o:
        for line in tqdm(o, total=num_word):
            array = line.lstrip().rstrip().split(" ")
            w = array[0]
            if w.lower() in word_set:
                vec = list(map(float, array[1:]))
                w2v[w.lower()] = vec
                w2i[w.lower()] = len(w2i)

    return w2v,w2i



def get_index_of_start_stop(context,x_w,start,stop):
    si_start, wi_start = get_si_and_wi(context,x_w,start)
    si_stop, wi_stop = get_si_and_wi(context,x_w,stop)
    return ((si_start,wi_start),(si_stop,wi_stop))

def get_si_and_wi(context, x_w, ci):
    cnt_dot = context[:ci].count(".")
    last_dot = context[:ci].rfind(".")
    front_words = word_tokenize(context[last_dot+1:ci])
    si_ci = cnt_dot
    wi_ci = len(front_words)
    return (si_ci,wi_ci)

def pre_prop_all():

    pre_process_src('train', out_name='total')
    pre_process_src('train', to_ratio=args['train_ratio'], out_name='train')
    pre_process_src('train', from_ratio=args['train_ratio'], out_name='dev')
    pre_process_src('dev', out_name='test')

def pre_process_src(src_name,from_ratio=0.0,to_ratio=1.0,out_name=None):
    # squad_train_dir = os.path.join(args.squad_dir, "train-v1.1.json")
    # squad_dev_dir = os.path.join(args.squad_dir, "dev-v1.1.json")
    #
    # train_data = json.load(open(squad_train_dir,'r'))
    # dev_data = json.load(open(squad_train_dir,'r'))

    src_dir = os.path.join(args['squad_dir'], "{}-v1.1.json".format(src_name))
    src_data = json.load(open(src_dir,'r'))

    from_a = int(round(len(src_data['data']) * from_ratio))
    to_a = int(round(len(src_data['data']) * to_ratio))

    x_aw = []
    x_ac = []

    qs_w = []
    qs_c = []
    x_i = []
    y = []
    y_ans = [] # list of answers
    word_set = set()
    # char_set = set()
    # text_data = []
    w2i_dict={}
    c2i_dict={}





    for a_i, article in enumerate(tqdm(src_data['data'][from_a:to_a])):

        x_pw = []
        x_pc = []
        x_aw.append(x_pw)
        x_ac.append(x_pc)
        for p_i, paragraph in enumerate(article['paragraphs']):

            context = paragraph['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            x_cw = list(map(word_tokenize, sent_tokenize(context)))
            x_cc = [[list(w) for w in sent] for sent in x_cw]
            # xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            x_pw.append(x_cw)
            x_pc.append(x_cc)
            """
            embed word list
            char : 전부 사용되다고 가정.
            word : 사용된 단어 카운트 후 그 단어들만 dict 에 추가.
            """

            for sent in x_cw:
                for w in sent:
                    word_set.add(w.lower())


                    #     char_set.add(c)


            for qnai, qna in enumerate(paragraph['qas']):
                """
                embed qna list
                """

                q_w = word_tokenize(qna['question'])
                q_c = [list(w) for w in q_w]

                for w in q_w:
                    word_set.add(w.lower())

                ys = []
                anss = []

                for ans in qna['answers']:
                    ans_start_c = ans['answer_start']
                    ans_stop_c = ans_start_c + len(ans['text']) - 1
                    ans_start_w, ans_stop_w = get_index_of_start_stop(context,x_cw,ans_start_c,ans_stop_c)

                    anss.append(ans['text'])
                    ys.append((ans_start_w,ans_stop_w))

                qs_w.append(q_w)
                qs_c.append(q_c)
                x_i.append((a_i,p_i))
                y.append(ys)
                y_ans.append(anss)

    w2v_dict,w2i_dict = get_w2v_dict(word_set)

    # i2v_dict = {w2i_dict[word]: vec for i, (word, vec) in enumerate(w2v_dict.items()) if word in w2i_dict}

    c_list = list('abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=<>()[]{}')
    for i,alphabet in enumerate(c_list):
        c2i_dict[alphabet] = i
    data = {
        'qs_w' : qs_w, # questions words' list # example : [['what','was','the','title','?'], .. , []]
        'qs_c' : qs_c, # questions chars' list # example : [[['w','h','a','t'],['w','a','s'],['t','h','e'],['t','i','t','l','e'],['?']], .. ,[]]
        'ans' : y_ans, # answers text list     # example : ["Whales","1994", ..]
        'x_i' : x_i,   # (article index, paragraph index) # example : [(3,10), ..]
        'y' : y,       # answers start&stop index # example : [((3,0),(4,10)), .. ,()]
    }

    fixed_data = {
        'w2v_dict': w2v_dict, # {'the' : [,,,,...,,], ',' : [,,,,...,,], ...}
        'w2i_dict': w2i_dict,
        'c2i_dict': c2i_dict,
        'x_aw': x_aw,
        'x_ac': x_ac
    }

    data_path = os.path.join(args['squad_dir'], "data_{}.json".format(out_name))
    fixed_data_path = os.path.join(args['squad_dir'], "fixed_data_{}.json".format(out_name))
    json.dump(data, open(data_path, 'w'))
    json.dump(fixed_data, open(fixed_data_path, 'w'))



if __name__ == "__main__":
    # pre_process_sec("dev")
    # pre_process_sec("train")
    # pre_process_sec("test")

    pre_prop_all()
