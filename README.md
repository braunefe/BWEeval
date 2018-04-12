### Evaluating bilingual word embeddings on the long tail

TODO

#### Data

TODO

#### Requirements

* Python 2.7
* dependencies in requirements.txt

```sh
	pip install -r requirements.txt
```

#### Run

* To reproduce the numbers in the paper run __make__
	* run make again to print the table
	* use -j n to run on n threads
	* use DEVICE=gpu to run maxmarg on GPU
	* runtime on cpu with 4 threads around 1 day

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
