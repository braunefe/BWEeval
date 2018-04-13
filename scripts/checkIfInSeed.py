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
    parser.add_option("-d", "--data", dest="data") #eval dict (test set)
    parser.add_option("-n", "--nbest", dest="nbest") #mined n-best lists (minedOOV)
    parser.add_option("-f", "--first", dest="output1") #output if right word is in 5-best
    parser.add_option("-s", "--second", dest="output2") #output if right word is in 1-best
 
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':

    options = get_options()

    #open output files
    out1 = open(options.output1, 'w')
    out2 = open(options.output2, 'w')		

    #read in seed lexicon and put into dictionary
    seed = open(options.data, 'r')
    d = {}

    #read nbest neighbors
    nbest = open(options.nbest, 'r')

    for line in seed: #pairs in seed lexicon
	pairs = line.split()
	d[pairs[0].strip()]=pairs[1].strip()


    for line in nbest:
	fields = line.split("|||")
	source = fields[0].strip()
        if source in d:
		i = 1
		found = 0
		# check if first item in ranked list is the correct target
		if len(fields) > 1:
			targets1 = fields[1].split();
			if len(targets1) > 0:				
				target1 = targets1[0].strip()
				if d[source] == target1: # check if right target is first best
					out2.write('{} {} 1 1\n'.format(source,target1))
				else:
					out2.write('{} {} 0 {}\n'.format(source,target1,d[source]))
			else:
				out2.write('{} has no target\n'.format(source))
		
		# check one of the items in 10-best is the correct target
		while i < len(fields):
			targets = fields[i].split()
			if len(targets) > 0:
				target = targets[0].strip()
				if d[source] == target:
					out1.write('{} {} 1 {}\n'.format(source,target,i))
					found = 1
			i = i+1
		if found == 0:
			out1.write('{} {} 0 {}\n'.format(source,target,d[source]))
			
