import collections
from collections import Set
import sys
import random

#read file containing target vocabulary
filename1 = sys.argv[1]

words=[]
for line in open(filename1).readlines():
        ws = line.split()
        for w in ws:
                words.append(w)

c = collections.Counter(words).most_common(500000)

unigramList=[]
listSize=300000000

allFreqs=0.0
for w in c: #table containing 300000 most frequent with frequencies                                                                                                           
        allFreqs=allFreqs+w[1]

probas=dict()
for w in c:
        i=0
        #j=round(w[1]/float(allFreqs))*listSize                                                                                                                                
        j= w[1]/float(allFreqs) * float(listSize)
        #print("J is {} for {}".format(j,w[0]))                                                                                                                                
        while i < j:
                unigramList.append(w[0])
                i=i+1

#file containing seed lexicon
filename2 = sys.argv[2]
file2 = open(filename2,'r')

filename3 = sys.argv[3]
out3 = open(filename3,'w')

filename4 = sys.argv[4]
out4 = open(filename4,'w')

filename5 = sys.argv[5]
out5 = open(filename5,'w')

amount_negatives = int(sys.argv[6])
print("Amount Negatives {}".format(amount_negatives))

#create headers in output files
#out3.write("s,l,n\n")
#out4.write("s\n")
#out5.write("t\n")

#put elements of seed lexicon into dictionary
d={}
for line in file2: #pairs in seed lexicon
	pairs = line.split()
	d[pairs[0].strip()]=pairs[1].strip()

#randomly select negative examples and print out
#set containing word types for source and target vocabulary
targetTypes = set()
sourceTypes = set()

for key,value in d.iteritems():
	#ignore columns for compatibility with pandas
	if key != "," and "," not in key and value != "," and "," not in value :
                sourceWords = []
                targetPositives = []
                targetNegatives = []
                drawnNegatives = set()
                i=0
		while i < amount_negatives:
			#print ("I {}, NEG {}".format(i,amount_negatives))
			neg=random.choice(unigramList)
                        while neg in drawnNegatives:
                                neg=random.choice(unigramList)
			if neg != value and neg != "," and "," not in neg: #select negative (value is positive instance)
                                targetTypes.add(value)
                                #targetPositives.append(value)
				targetTypes.add(neg)
                                targetNegatives.append(neg)
                                drawnNegatives.add(neg)
				sourceTypes.add(key)
                                #sourceWords.append(key)
                                #out3.write("{},{},{}\n".format(key,value,neg)
                                i=i+1
                so = key
                tp = value
                tn = " ".join(targetNegatives)
                drawnNegatives.clear()
                out3.write("{},{},{}\n".format(so,tp,tn))
                #print("LENGTH OF NEGATIVES: {}".format(len(targetNegatives)))

for s in sourceTypes:
	out4.write("{}\n".format(s.strip()))

for t in targetTypes:
	out5.write("{}\n".format(t.strip()))

file2.close()
out3.close()
out4.close()
out5.close()		


