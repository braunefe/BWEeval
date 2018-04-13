### Evaluating bilingual word embeddings on the long tail

Bilingual word embeddings are useful for bilingual lexicon induction,
the task of mining translations of given words. Many studies have
shown that bilingual word embeddings perform well for bilingual
lexicon induction but they focused on frequent words in general
domains. For many applications, bilingual lexicon induction of rare
and domain-specific words is of critical importance. Therefore, we
design a new task to evaluate bilingual word embeddings on rare words
in different domains. We show that state-of-the-art approaches fail on
this task and present simple new techniques to improve bilingual word
embeddings for mining rare words. We release new gold standard datasets
and code to stimulate research on this task.

#### Data

* data/lexicon
	* train\* are the lexicons for training post-hoc mapping
	* validation\*/EvalDict\* are the validation/test sets
	* *(we call HIML the medical datasets)*
* external data
	* run *make get_data* to download data only ([link](http://cis.uni-muenchen.de/~hangyav/data/rareword_mining.zip))
	* data/embeddings: pretrained embeddings
	* data/texts: raw texts used for the experiments

#### Requirements

* Python 2.7
* dependencies in requirements.txt

```sh
	pip install -r requirements.txt
```

#### Run

* To reproduce the numbers in the paper run __make__
	* use -j n to run on n threads
	* use DEVICE=gpu to run maxmarg on GPU
	* runtime on cpu with 4 threads around 1 day
	* run make again to print the table with results

```
	make -j 4 DEVICE=cpu
	make
```



#### Citation

```
@inproceedings{artetxe2017learning,
	author    = {Braune, Fabienne  and  Hangya, Viktor  and  Eder, Tobias and Fraser, Alexander},
	title     = {Evaluating bilingual word embeddings on the long tail},
	booktitle = {Proceedings of the 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics},
	year      = {2018},
	pages     = {1--6}
}
```
