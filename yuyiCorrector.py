#!/usr/bin/env python
# encoding: utf-8
# @Time    : 2019/5/7 15:29
# @Author  : lxx
# @File    : yuyiCorrector.py
# @Software: PyCharm
import kenlm
import jieba


all_train=[]
for line in open("data/data.train",encoding="utf-8"):
    line=line.strip()
    line=line.split("\t")
    sentens=line[2:]
    if len(sentens) >=2:
        for i in range(1,len(sentens)):
            tmp = []
            tmp.append(sentens[0])
            tmp.append(sentens[i])
            all_train.append(tmp)
    else:
        all_train.append([sentens[0],sentens[0]])

print(len(all_train))


lm = kenlm.Model("kenlm-model/people_chars_lm.klm")
err_count=0
for pair_sen in all_train:
    sen_seg0=jieba.cut(pair_sen[0])
    sen_seg1=jieba.cut(pair_sen[1])

    ppl_score0=lm.perplexity(' '.join(sen_seg0))
    ppl_score1=lm.perplexity(' '.join(sen_seg1))
    if ppl_score1 > ppl_score0:
        err_count+=1
print(err_count)
print(len(all_train)-err_count)
print("222222222222222222222222222222222222222222")
print("yuyi")
print("github")
