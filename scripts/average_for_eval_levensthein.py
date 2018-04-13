#!/usr/bin/env python
import logging
from optparse import OptionParser
from collections import defaultdict
import sys
from sets import Set
import numpy as np
import itertools
import Levenshtein


#Average distances for different numeric values
def get_options():
    parser = OptionParser()
    parser.add_option("-d","--dist", action="append", dest="my_dists") #list of distances to average
    parser.add_option("-o", "--output", dest="output") #output of average weighted distances
    parser.add_option("-v", "--eval", dest="eval") #validation set to evaluate against
    parser.add_option("-p", "--params", action="append", dest="params") #hyperparameters
    parser.add_option("-e", "--edit", dest="edit") #hyperparameters

    (options, args) = parser.parse_args()
    return options


def load_valid(validFile):
    #read in valid lexicon and put into dictionary
    valid = open(validFile, 'r')
    d = {}

    #read nbest neighbors
    nbest = open(validFile, 'r')

    for line in valid: #pairs in valid lexicon
    	pairs = line.split()
    	d[pairs[0].strip()]=pairs[1].strip()
    
    return d


def average_n_distances(hyperparams,editParam,distFiles,validFile): #average distances for list of hyperparameters

    #open vectors to add into dict
    distance_average = defaultdict(lambda: defaultdict(float))

    for n, entry in enumerate(distFiles):
    	with open(entry, 'r') as file:
    		for line in file:
    			fields = line.split('|||')
    			dictentry = fields[0].strip()
    			i = 1
    			while i < len(fields):
    				field = fields[i].split()
    				if len(field) > 0:
    					wordentry = field[0].strip()
    					numentry = float(field[1].strip())
    					distance_average[dictentry][wordentry] += float(hyperparams[n]) * numentry
    				i+=1

    for source_word, dictionary in distance_average.items():
        for key, value in dictionary.items():
            dictionary[key] = (value + float(editParam) * (Levenshtein.ratio(unicode(source_word, "utf8"), unicode(key, "utf8")))) / float(len(options.my_dists)+1) 

    d=load_valid(validFile)
    
    correctFirst = 0
    correctN = 0
    allWords = len(d.keys())

	#EVALUATE AVERAGED LIST AGAINST VALIDATION SET
    for word in distance_average.keys(): #source word
    	keyorder = sorted(distance_average[word], key=distance_average[word].get, reverse=True) #sorted average distances

    	#find source in validation dictionary
    	if word in d: #found source word in validation dict
    		if d[word] == keyorder[0]:
    			correctFirst = correctFirst+1
    			correctN = correctN+1
    		elif d[word] != keyorder[0] and d[word] in keyorder[:5]:
    			correctN = correctN+1

#return eval numbers
    h=options.params
    h.append(options.edit)
    h.append(str(correctFirst))
    h.append(str(correctN))
    h.append(str(allWords))
    return h

def print_eval(t):
	s=","
	print(s.join(list(map(str,t))))

options=get_options()


print("h1,h2,h3,h4,e,1-best,n-best,all") #print header of csv file
r = average_n_distances(options.params,options.edit,options.my_dists,options.eval)
print_eval(r)#pass number of hyperparameters






