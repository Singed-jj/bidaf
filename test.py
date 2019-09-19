import nltk
sent_tokenize = nltk.sent_tokenize
def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

context = "The state also has five Micropolitan Statistical Areas centered on Bozeman, Butte, Helena, Kalispell and Havre. These communities, excluding Havre, are colloquially known as the \"big 7\" Montana cities, as they are consistently the seven largest communities in Montana, with a significant population difference when these communities are compared to those that are 8th and lower on the list. According to the 2010 U.S. Census, the population of Montana's seven most populous cities, in rank order, are Billings, Missoula, Great Falls, Bozeman, Butte, Helena and Kalispell. Based on 2013 census numbers, they collectively contain 35 percent of Montana's population. and the counties containing these communities hold 62 percent of the state's population. The geographic center of population of Montana is located in sparsely populated Meagher County, in the town of White Sulphur Springs."
xi = list(map(word_tokenize, sent_tokenize(context)))

cxi = [[list(xijk) for xijk in xij] for xij in xi]
# print(xi)
# print("==========")
# print(cxi)
qi = word_tokenize("this is an example statement")
print(qi)



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



spanss = get_2d_spans(context,xi)
idxs = []

start = 381
stop = 464
for sent_idx, spans in enumerate(spanss):
    for word_idx, span in enumerate(spans):
        if not (stop <= span[0] or start >= span[1]):
            idxs.append((sent_idx, word_idx))
assert len(idxs) > 0, "{} {} {} {}".format(context, spanss, start, stop)
res = idxs[0], (idxs[-1][0], idxs[-1][1] + 1)
print("spanss")
print(spanss)
print("res : ",res)
