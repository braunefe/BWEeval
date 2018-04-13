import logging
from gensim.models import Word2Vec
import pandas as pd
from optparse import OptionParser
from sklearn.linear_model import Ridge
import pickle
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

def get_options():
    parser = OptionParser()
    parser.add_option("-d", "--data", dest="data") #lexicon (seeds), tab separated, (header with fist line: en column 1, de column 2. (en tab sp)
    parser.add_option("-f", "--from", dest="fr", default='en') #map from input
    parser.add_option("-t", "--to", dest="to", default='de') #to output
    parser.add_option("--po", "--projection_output", dest="pred_out")#mapped w2v #from here leave
    parser.add_option("--mo", "--model_output", dest="model_out")#regression model (if no param no save)
    parser.add_option("--fv", "--from_vectors", dest="from_vectors")#en w2v
    parser.add_option("--tv", "--to_vectors", dest="to_vectors")#de w2v

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    options = get_options()

    fr = options.fr
    to = options.to
    data = pd.read_csv(options.data, sep='\t')
    fw2v = Word2Vec.load_word2vec_format(options.from_vectors, binary=False)
    tw2v = Word2Vec.load_word2vec_format(options.to_vectors, binary=False)

    data['contains'] = data.en.apply(fw2v.__contains__)

    logging.info('OOV: {}'.format(1.0 - float(len(data[data.contains]))/len(data)))

    tmp = [(fw2v[row[fr]], tw2v[row[to]]) for i, row in data.iterrows() if row[fr] in fw2v and row[to] in tw2v]

    X = [e[0] for e in tmp]
    Y = [e[1] for e in tmp]

    logging.info('Training...')
    model = Ridge()
    model.fit(X, Y)

    if options.model_out is not None:
        logging.info('Saving model to: {}'.format(options.model_out))
        with open(options.model_out, 'wb') as fout:
            pickle.dump(model, fout)

    if options.pred_out is not None:
        logging.info('Saving projection to: {}'.format(options.pred_out))
        with open(options.pred_out, 'w') as fout:
            fout.write('{} {}\n'.format(len(fw2v.vocab), tw2v.vector_size))
            for w in fw2v.vocab:
                fout.write('{} {}\n'.format(w, ' '.join([str(v) for v in model.predict([fw2v[w]])[0]])))
