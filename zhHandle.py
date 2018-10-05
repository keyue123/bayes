#!/usr/bin/python                                                                                                                                                                                                                    
#coding=utf-8

#File Name: test.py
#Author   : john
#Mail     : john.y.ke@mail.foxconn.com 
#Created Time: Sat 01 Sep 2018 05:38:56 PM CST

import bayes
import jieba
import pandas as pd
from numpy import *

# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 对句子进行分词
def wordCut(sentence):
    words = jieba.cut(sentence.strip())
    stopwords = stopwordslist('C:\\Users\\John\\Desktop\\emotion Analysis\\stopKeyWords.txt')  # 这里加载停用词的路径
    outstr = []
    for word in words:
        if word not in stopwords:
            if word != '\t':
                outstr.append(word)
    return outstr

def DataHandle(filename, flag):
    out = []
    #lines = pd.read_table("C:\\Users\\John\\Desktop\\emotion Analysis\\goods.txt", header=None, encoding='utf-8', names=['评论'])
    lines = pd.read_table(filename, header=None, encoding='utf-8', names=['评论'])
    for line in lines['评论']:
        line = str(line)
        outStr = wordCut(line)  # 这里的返回值是字符串
        out.append(outStr)

    if flag:
        Vec = [1] * lines.shape[0]
    else:
        Vec = [0] * lines.shape[0]

    return Vec, out

if __name__ == '__main__':
    googDataPath = 'C:\\Users\\John\\Desktop\\emotion Analysis\\goods.txt'
    badDataPath = 'C:\\Users\\John\\Desktop\\emotion Analysis\\bad.txt'

    # 1 好评     0 差评
    goodVec, goodList = DataHandle(googDataPath, 1)
    badVec, badList = DataHandle(badDataPath, 0)

    listClasses = goodVec + badVec
    listOPosts = goodList + badList
    print(listClasses)
    print(listOPosts)

    myVocabList = bayes.createVocabList(listOPosts)
    print(myVocabList)
    # 3. 计算单词是否出现并创建数据矩阵
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
    # 4. 训练数据
    p0V, p1V, pAb = bayes.trainNB0(array(trainMat), array(listClasses))
    # 5. 测试数据
    while True:
        inputS = input(u'请输入您对本商品的评价：')

        testEntry = wordCut(inputS)
        thisDoc = array(bayes.setOfWords2Vec(myVocabList, testEntry))
        print('评价: ', bayes.classifyNB(thisDoc, p0V, p1V, pAb))