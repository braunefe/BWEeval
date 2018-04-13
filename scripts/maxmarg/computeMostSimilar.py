#!/usr/bin/env python
import logging
from gensim.models import Word2Vec
import pandas as pd
from optparse import OptionParser
import numpy as np
from scipy.spatial import distance
from scipy import stats
from collections import defaultdict
import sys
from sets import Set


def get_options():
    parser = OptionParser()
    parser.add_option("-s", "--source", dest="source") #list of source words from which we extract 10 closest neighbors in BWE
    parser.add_option("-t", "--target", dest="target") #Embedding of target language words (embedding.de)
    parser.add_option("-v", "--vectors", dest="vectors") #BWE of source words mapped into target space
    parser.add_option("-o", "--output", dest="output") #output file
 
    (options, args) = parser.parse_args()
    return options

def get_most_similar(word, f, t):
    if word in f:
        return t.similar_by_vector(f[word],topn=5)
    else:
        print "{} not in embedding".format(word)
        return t.similar_by_vector(np.zeros(f.vector_size),topn=5)

if __name__ == '__main__':

    options = get_options()

    #open output files
    out1 = open(options.output, 'w')

    #load BWE of source words projected into target space
    w2v = Word2Vec.load_word2vec_format(options.vectors, binary=False)

    #load MWE of target words (in target space)
    tw2v = Word2Vec.load_word2vec_format(options.target, binary=False)

    #read in a list of source words
    sourceFile = open(options.source)
    sourcewords = sourceFile.readlines()

    #print Targets
    for word in sourcewords:
        rankedList = get_most_similar(word.strip(),w2v,tw2v)
        out1.write('{} |||'.format(word.strip()))
        for i in rankedList:
            a = i[0].encode('utf-8')
            out1.write('{} {} |||'.format(a,i[1]))
        out1.write('\n')						
