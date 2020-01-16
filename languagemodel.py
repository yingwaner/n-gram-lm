from collections import Counter
import numpy as np
import math


#ppl for bigram
def ppl_bigram(sentence, word2id, unigram, bigram):
    sentence = sentence.strip().split()
    s = [word2id[w] for w in sentence]
    les = len(s)
    if les < 1:
        return 0
    p = unigram[s[0]]
    if les < 2:
        return p
    for i in range(1, les):
        p *= bigram[s[i - 1], s[i]]
    ppl = math.pow(1/p, 1.0/les)
    return ppl

#ppl for unigram
def ppl_unigram(sentence, word2id, unigram):
    sentence = sentence.strip().split()
    s = [word2id[w] for w in sentence]
    les = len(s)
    if les < 1:
        return 0
    p = unigram[s[0]]
    for i in range(1, les):
        p *= unigram[s[i]]
    ppl = math.pow(1/p, 1.0/les)
    return ppl


#Add one smoothing
def add_one(ngram):
    ngram += 1
    return ngram

def good_tune_unigram(unigram):
    uni = unigram.tolist()
    Num = getNUM(uni)
    for i in range(len(uni)):
        if unigram[i] == 0:
            unigram[i] = Num[1]
        else:
            if unigram[i]+1 in Num:
                unigram[i] = (unigram[i] + 1) * Num[unigram[i]+1] / Num[unigram[i]]
            else:
                unigram[i] = (unigram[i] + 1) / Num[unigram[i]]
    return unigram

#get sum for each word frequency
def getNUM(uni):
    Num = {}
    for sth in uni:
        Num.setdefault(sth, 0)
        Num[sth] += 1
    return Num

def main():
    #prepare corpus
    corpus_file = open('ptb.valid.txt')
    corpus = corpus_file.readlines()
    test = open('ptb.test.txt').readlines()

    #corpus preprocess
    counter = Counter()  # 词频统计
    for sentence in corpus:
        sentence = sentence.strip().split()
        for word in sentence:
            counter[word] += 1
    counter = counter.most_common()
    lec = len(counter)
    word2id = {counter[i][0]: i for i in range(lec)}
    id2word = {i: w for w, i in word2id.items()}


    """N-gram modeling"""
    unigram = np.array([i[1] for i in counter])
    
    #using good-tune smoothing for unigram
    unigram = good_tune_unigram(unigram)
    unigram = unigram / sum(i[1] for i in counter)
    
    bigram = np.zeros((lec, lec)) + 1e-8
    for sentence in corpus:
        sentence = sentence.strip().split()
        sentence = [word2id[w] for w in sentence]
        for i in range(1, len(sentence)):
            bigram[[sentence[i - 1]], [sentence[i]]] += 1
    #using add one smoothing for bigram
    bigram = add_one(bigram)
    for i in range(lec):
        bigram[i] /= bigram[i].sum()
    np.save("unigram.npy", unigram)
    np.save("bigram.npy", bigram)
    """语料较大，trigram的会爆内存，运行不了，但是我觉得代码没问题
    trigram = np.zeros([lec, lec, lec]) + 1e-8
    for sentence in corpus:
        sentence = sentence.strip().split()
        sentence = [word2id[w] for w in sentence]
        for i in range(2, len(sentence)):
            trigram[sentence[i - 2], sentence[i-1], sentence[i]] += 1
    for i in range(lec):
        for j in range(lec):
            trigram[i, j] /= trigram[i, j].sum()
    """
    """
    #output the ppl for unigram
    for i in range(5):
        print(test[i].strip(), ppl_unigram(test[i].strip(), word2id, unigram))

    #ouput the ppl for bigram
    for i in range(5):
        print(test[i].strip(), ppl_bigram(test[i].strip(), word2id, unigram, bigram))
    """

if __name__ == '__main__':
    main()

