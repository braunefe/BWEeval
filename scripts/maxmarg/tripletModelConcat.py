from __future__ import print_function

import pandas as pd
import numpy as np
import logging
import sys

from theano import function, printing
from theano import tensor as T
from keras import backend as K
from keras.callbacks import History, EarlyStopping 
from keras.models import Sequential, Model, model_from_yaml
from keras.layers import Embedding, Flatten, Input, merge, Dense, Reshape, RepeatVector, Lambda, Merge
from keras.optimizers import Adam, Adagrad
from optparse import OptionParser
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import fasttext
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
sys.setrecursionlimit(5000)

def get_options():

    parser = OptionParser()

    parser.add_option("-t", "--train", dest="train") #train file (comma separated, header, id in first column, column-named text,  column-named label)
    parser.add_option("-e", "--test", dest="test") #test file (comma separated, header)
    parser.add_option("-n", "--negatives", dest="negatives") #negative example file (comma separated, header)
    parser.add_option("--so", "--source", dest="source") #source language file
    parser.add_option("--ta", "--target", dest="target") #target language file
    parser.add_option("--sv", "--source_vectors", dest="source_vectors") #w2v vectors -> BWE in same file
    parser.add_option("--tv", "--target_vectors", dest="target_vectors") #w2v vectors -> BWE in same file
    parser.add_option("--ev", "--test_vectors", dest="test_vectors") #(w2v vectors (omit if BWE) -> concatenate and use 1 file (1 for en, 1 for es)
    parser.add_option("-o", "--output", dest="output", default='model') #save out model -> outcommented
    parser.add_option("--val_split", dest="val_split", default=0.2) #set float for validation (what should be used as valid)
    parser.add_option("--ne", dest="epoch", default=25, type=int) #nbr epochs
    parser.add_option("-b", "--batch", dest="batch", default=50, type=int) #batch size
    parser.add_option("--nbneg", "--nbneg", dest="nbneg", default=50, type=int) #number of negatives
    parser.add_option("-m", "--margin", dest="margin", default=1, type=int) #batch size
    parser.add_option("--vt", "--vector-type", dest="vector_type", default='txt', type=str) #text version (bin if binary vector)
    parser.add_option("--pr", "--proj", dest="proj", default='proj') #text version (bin if binary vector)

    (options, args) = parser.parse_args()
    return options

class MaxMargin():
    def __init__(self, num_negatives, margin):
        self.num_negatives = num_negatives
        self.margin = margin

    def __call__(self, X):
        source,positive,negative = X

        #normalize vectors to compute cosine distance with dot product
        source_norm = K.l2_normalize(source,axis=1)
        positive_norm = K.l2_normalize(positive,axis=1)

        #transform cosine similarity into distance
        pos = np.arccos(K.dot(source_norm,K.transpose(positive_norm)))
    
        negative_norm=K.l2_normalize(negative[:,0],axis=1)
        neg = np.arccos(K.dot(source_norm,K.transpose(negative_norm)))

        #for each negative example compute max margin
        negScore = K.maximum(0.0, self.margin + pos - neg)
        for i in range(1, self.num_negatives):
            negative_norm=K.l2_normalize(negative[:,i],axis=1)
            neg = np.arccos(K.dot(source_norm,K.transpose(negative_norm)))
            negScore = negScore + K.maximum(0.0, self.margin + pos - neg)
        
        return negScore
     

def save_model(model, out):
    logging.info('Saving model to....')
    yaml_string = model.to_yaml()
    with open('{}.yaml'.format(out), 'w') as fout:
        fout.write(yaml_string)
        logging.info('\t{}.yaml'.format(out))
    model.save_weights('{}.weights'.format(out))
    logging.info('\t{}.weights\n'.format(out))

def load_model(name):
    logging.info('Loading model {}...'.format(name))
    with open('{}.yaml'.format(name), 'r') as yaml_file:
        loaded_model_yaml = yaml_file.read()
    loaded_model = model_from_yaml(loaded_model_yaml)
    yaml_file.close()

    loaded_model.load_weights('{}.weights'.format(name))
    return loaded_model

def load_triplets(train, target, nbneg, test=0.25, maxlen=None):
    
    logging.info('Loading triplets...')
    
    trd = pd.read_csv(train,header=None,names=['s', 'l', 'n'],quoting=3)
    tv = pd.read_csv(target,header=None,names=['t'],quoting=3)
    trd.s=trd.s.astype(str)
    trd.l=trd.l.astype(str)
    trd.n=trd.n.astype(str)
    tv.t=tv.t.astype(str) 


    #fit all the source words : each word gets an index
    s = Tokenizer(filters=",")

    # fit all target words : each word gets an index
    t = Tokenizer(filters=",")
    
    # FIT TRAINING AND TEST WORDS ON TEXT AND RESTORE INTO TRD
    s.fit_on_texts(trd.s)
    t.fit_on_texts(tv.t.tolist() + list({w for l in trd.n for w in l.split()}))

    so_seqs = s.texts_to_sequences(trd.s)
    tp_seqs = t.texts_to_sequences(trd.l)
    tn_seqs = t.texts_to_sequences(trd.n)
    
    trd['source'] = so_seqs
    trd['positive'] = tp_seqs
    trd['negative'] = tn_seqs

    nb_tw = len(t.word_index.items()) + 1
    nb_sw = len(s.word_index.items()) + 1
    seq_len = len(tn_seqs[0])

    if type(test) == float:
        ted = trd.tail(n=int(float(trd.shape[0])*test))
        trd = trd[~trd.index.isin(ted.index)]
    else: #TODO IF TEST IS GIVEN AS ARGUMENT
        ted = pd.read_csv(test)
    
    return trd, ted, s, t, nb_sw, nb_tw, seq_len

#Identity loss for a batch
def identity_loss(y_true, y_pred):   
    return K.abs(K.sum(y_pred - 0 * y_true))

def build_model(nb_source_words, nb_target_words, embedding_dim=300, embedding_dropout=0.0, static_embedding=True, source_tokenizer=None, target_tokenizer=None,
                 source_w2v=None,target_w2v=None,nb_neg=None,margin=1.0):

    logging.info('Building model ...')

    #Create embedding weights for source and target space
    #compute embedding of source word in source vector space
    if source_tokenizer and type(dict()) == dict:
        source_embedding_weights = np.zeros((nb_source_words, embedding_dim))
        for w,i in source_tokenizer.word_index.items(): #word-index pairs in tokenizer
            #print('word index {} {}'.format(w,i))
            if type(source_w2v) == fasttext.model.WordVectorModel or w in source_w2v: #ask Vitkor
                source_embedding_weights[i, :] = source_w2v[w]
            else:
                source_embedding_weights[i, :] = np.random.uniform(-0.25, 0.25, embedding_dim)

    #compute embedding of target word in target vector space
    if target_tokenizer and type(dict()) == dict:
        target_embedding_weights = np.zeros((nb_target_words, embedding_dim))
        for w,i in target_tokenizer.word_index.items(): #word-index pairs in tokenizer
            if type(target_w2v) == fasttext.model.WordVectorModel or w in target_w2v: #ask Vitkor
                target_embedding_weights[i, :] = target_w2v[w]
            else:
                target_embedding_weights[i, :] = np.random.uniform(-0.25, 0.25, embedding_dim)

    # Input one-hot vectors
    source_item = Input(shape=(1,))
    positive_item = Input(shape=(1,))
    negative_items = Input(shape=(nb_neg,))


    #branch dealing with source words
    #monolingual semantic embedding
    source_embedding_layer = Embedding(nb_source_words, embedding_dim, input_length=1, dropout=embedding_dropout, trainable=(not static_embedding), weights=None if source_embedding_weights is None else[source_embedding_weights],name="source_embedding")(source_item)
    #layer that predicts projection
    source_dense_layer = Dense(embedding_dim, input_dim=embedding_dim,name="source_dense")(source_embedding_layer)
    source = Flatten()(source_dense_layer)

    #branch dealing with target words
    #layer embedding target words
    positive_item_embedding = Embedding(nb_target_words, embedding_dim, input_length=1, dropout=embedding_dropout,
            trainable=(not static_embedding), weights=None if target_embedding_weights is None
            else[target_embedding_weights],name="positive_embedding")(positive_item)
    positive = Flatten()(positive_item_embedding)

    negative_item_embedding = Embedding(nb_target_words, embedding_dim, input_length=nb_neg, dropout=embedding_dropout,
            trainable=(not static_embedding), weights=None if target_embedding_weights is None
            else[target_embedding_weights],name="negative_embedding")(negative_items)


    merged = Merge(mode=MaxMargin(nb_neg, margin).__call__, output_shape=(1,))([source,positive,negative_item_embedding])

    model = Model(
            input=[source_item, positive_item, negative_items],output=merged)
    model.compile(loss=identity_loss, optimizer='adagrad')

    return model


if __name__ == '__main__':

    options = get_options()

    embedding_dim = 300

    #Get the margin right (input is just an ordered list of int)
    marg = float(options.margin)*0.1 #increase margin by 0.1 steps

    print(marg)

    # Read data
    train, test, stok, ttok, nb_s, nb_t, sl = load_triplets(options.train, options.target, options.nbneg, test=0.25, maxlen=None)
    

    #Replace each word by its index
    train_s = np.asarray(train.source.tolist())
    train_p = np.asarray(train.positive.tolist())
    train_n = np.zeros((train.negative.shape[0], sl))
    train_n = np.array(train.negative.tolist())
    test_s = np.asarray(test.source.tolist())
    test_p = np.asarray(test.positive.tolist())
    test_n = np.asarray(test.negative.tolist())

    print(train_s.shape)
    print(train_p.shape)
    print(train_n.shape)
    print(test_s.shape)
    print(test_p.shape)
    print(test_n.shape)


    # Load w2v models
    # Load source word embedding space
    if options.source_vectors is not None:
        if options.vector_type == 'fast':
            sw2v = fasttext.load_model(options.svectors)
        else:
            sw2v = Word2Vec.load_word2vec_format(options.source_vectors, binary=options.vector_type=='bin')
            if options.test_vectors is not None:
                sw2v = {w: sw2v[w] for w in sw2v.vocab}
                sw2v2 = Word2Vec.load_word2vec_format(options.test_vectors, binary=options.vector_type=='bin')
                for w in sw2v2.vocab:
                    sw2v[w] = sw2v2[w]


    # Load target word embedding space
    if options.target_vectors is not None:
        if options.vector_type == 'fast':
            tw2v = fasttext.load_model(options.tvectors)
        else:
            tw2v = Word2Vec.load_word2vec_format(options.target_vectors, binary=options.vector_type=='bin')
            if options.test_vectors is not None:
                tw2v = {w: tw2v[w] for w in tw2v.vocab}
                tw2v2 = Word2Vec.load_word2vec_format(options.test_vectors, binary=options.vector_type=='bin')
                for w in tw2v2.vocab:
                    tw2v[w] = tw2v2[w]


    #Build model
    model = build_model(nb_s, nb_t, source_tokenizer=stok, target_tokenizer=ttok,source_w2v=sw2v,target_w2v=tw2v,nb_neg=sl,margin=marg)

    # Print the model structure
    print(model.summary())

    logging.info('Training...')
    
    #manual early stopping
    p=10
    patience=[]
    prevLoss=100000
    for i in range(1,options.epoch):
    	model.fit([train_s,train_p,train_n],
    	np.zeros(len(train_s)),
    	batch_size=options.batch,
    	nb_epoch=1,
    	verbose=1,
    	shuffle=True)
    	testLoss = model.evaluate([test_s,test_p,test_n],np.zeros(len(test_s)),batch_size=options.batch)
    	print("EVAL {} {} {}".format(i,testLoss,len(patience)))
    	if(testLoss >= prevLoss):
    		patience.append(testLoss)
    		if(len(patience) > p):
    			break
    	else:
    		patience.sort()
                if len(patience)>0:
                    testLoss = patience[0]
                    patience=[]
    	prevLoss = testLoss

    #compute projection with this model
    mapping = Sequential()
    mapping.add(Dense(embedding_dim,input_shape=(1,embedding_dim),weights=model.get_layer("source_dense").get_weights(),activation="linear"))
    if options.proj is not None:
        logging.info('Saving projection to: {}'.format(options.proj))
        with open(options.proj, 'w') as fout:
            fout.write('{} {}\n'.format(len(sw2v.vocab), tw2v.vector_size)) #We assume that we project into a space of the same dimension
            for w in sw2v.vocab:
                vec=sw2v[w]
                vec=np.reshape(vec, (1,1,300))
                fout.write('{} {}\n'.format(w, ' '.join([str(v) for v in mapping.predict([vec])[0][0]])))
