PYTHON = python
BASH = bash -c

DIR_RESULTS = ./results
DIR_RESULTS_RIDGE = $(DIR_RESULTS)/ridge
DIR_RESULTS_MAXMARG = $(DIR_RESULTS)/maxmarg
DIR_RESULTS_MAXMARG_TRIPLETS = $(DIR_RESULTS_MAXMARG)/triplets
DIR_RESULTS_MAXMARG_PROJECTIONS = $(DIR_RESULTS_MAXMARG)/projections
DIR_RESULTS_ENSEMBLE = $(DIR_RESULTS)/ensemble
DIR_RESULTS_ENSEMBLE_RIDGE = $(DIR_RESULTS_ENSEMBLE)/ridge
DIR_RESULTS_ENSEMBLE_MAXMARG = $(DIR_RESULTS_ENSEMBLE)/maxmarg

DIR_SCRIPTS = ./scripts
DIR_SCRIPTS_RIDGE = $(DIR_SCRIPTS)/ridge
DIR_SCRIPTS_MAXMARG = $(DIR_SCRIPTS)/maxmarg

URL_DATA = http://cis.uni-muenchen.de/~hangyav/data/rareword_mining.zip
DIR_DATA = ./data
DIR_LEXICONS = $(DIR_DATA)/lexicons
DIR_EMBEDDINGS = $(DIR_DATA)/embeddings
DIR_EMBEDDINGS_GENERAL = $(DIR_EMBEDDINGS)/general
DIR_EMBEDDINGS_MEDICAL = $(DIR_EMBEDDINGS)/medical
DIR_TEXTS = $(DIR_DATA)/texts

LEXICON_TRAIN_GENERAL = $(DIR_LEXICONS)/train.GENERAL.txt
LEXICON_TRAIN_MEDICAL = $(DIR_LEXICONS)/train.HIML.txt
LEXICON_TEST_GENERAL = $(DIR_LEXICONS)/EvalDict.GENERAL.txt
LEXICON_TEST_CRAWL = $(DIR_LEXICONS)/EvalDict.CRAWL.txt
LEXICON_TEST_NEWS = $(DIR_LEXICONS)/EvalDict.NEWS.txt
LEXICON_TEST_HIML = $(DIR_LEXICONS)/EvalDict.HIML.txt
LEXICON_TEST_RAREHIML = $(DIR_LEXICONS)/EvalDict.RAREHIML.txt

NUM_NEGATIVES = 2000
NUM_MARG_GENERAL_SKIP = 8
NUM_MARG_GENERAL_CBOW = 8
NUM_MARG_GENERAL_FTSKIP = 13
NUM_MARG_GENERAL_FTCBOW = 9
NUM_MARG_HIML_SKIP = 8
NUM_MARG_HIML_CBOW = 12
NUM_MARG_HIML_FTSKIP = 8
NUM_MARG_HIML_FTCBOW = 9
NUM_MARG_NEWS_SKIP = 8
NUM_MARG_NEWS_CBOW = 8
NUM_MARG_NEWS_FTSKIP = 13
NUM_MARG_NEWS_FTCBOW = 9
NUM_MARG_CRAWL_SKIP = 8
NUM_MARG_CRAWL_CBOW = 8
NUM_MARG_CRAWL_FTSKIP = 12
NUM_MARG_CRAWL_FTCBOW = 9
NUM_MARG_RAREHIML_SKIP = 8
NUM_MARG_RAREHIML_CBOW = 12
NUM_MARG_RAREHIML_FTSKIP = 8
NUM_MARG_RAREHIML_FTCBOW = 9

# gamma weights for ensemble (skip, cbow, ftskip, ftcbow) separated by _
WEIGHTS_ENSEMBLE_RIDGE_GENERAL = 0.0_0.1_0.6_0.0
WEIGHTS_ENSEMBLE_RIDGE_NEWS = 0.0_0.1_0.9_0.0
WEIGHTS_ENSEMBLE_RIDGE_CRAWL = 0.0_0.1_0.9_0.0
WEIGHTS_ENSEMBLE_RIDGE_HIML = 0.1_0.1_0.3_0.8
WEIGHTS_ENSEMBLE_RIDGE_RAREHIML = 0.0_0.0_0.4_0.3
WEIGHTS_ENSEMBLE_MAXMARG_GENERAL = 0.1_0.1_0.8_0.1
WEIGHTS_ENSEMBLE_MAXMARG_NEWS = 0.0_0.0_1.0_0.1
WEIGHTS_ENSEMBLE_MAXMARG_CRAWL = 0.0_0.0_1.0_0.1
WEIGHTS_ENSEMBLE_MAXMARG_HIML = 0.1_0.1_0.9_0.1
WEIGHTS_ENSEMBLE_MAXMARG_RAREHIML = 0.2_0.0_0.7_0.3
# gamma weights for ensemble+edit (skip, cbow, ftskip, ftcbow, edit) separated by _
WEIGHTS_ENSEMBLE_EDIT_RIDGE_GENERAL = 0.0_0.2_0.9_0.0_0.1
WEIGHTS_ENSEMBLE_EDIT_RIDGE_NEWS = 0.0_0.1_0.7_0.0_0.1
WEIGHTS_ENSEMBLE_EDIT_RIDGE_CRAWL = 0.3_0.3_0.5_0.1_1.0
WEIGHTS_ENSEMBLE_EDIT_RIDGE_HIML = 0.3_0.3_0.6_0.1_1.0
WEIGHTS_ENSEMBLE_EDIT_RIDGE_RAREHIML = 0.3_0.3_0.3_0.1_1.0
WEIGHTS_ENSEMBLE_EDIT_MAXMARG_GENERAL = 0.1_0.1_0.8_0.0_0.1
WEIGHTS_ENSEMBLE_EDIT_MAXMARG_NEWS = 0.0_0.1_1.0_0.0_0.2
WEIGHTS_ENSEMBLE_EDIT_MAXMARG_CRAWL = 0.0_0.4_0.2_0.0_0.8
WEIGHTS_ENSEMBLE_EDIT_MAXMARG_HIML = 0.1_0.1_1.0_0.1_0.4
WEIGHTS_ENSEMBLE_EDIT_MAXMARG_RAREHIML = 0.0_0.1_0.1_0.1_1.0

DEVICE = cpu
THREADS = 5
METHODS = skip cbow ftskip ftcbow

define split
	$(word $(2),$(subst $(3), ,$(1)))
endef

define split_dot
	$(word $(2),$(subst ., ,$(1)))
endef

############################################################################
############################################################################

all: sep1 results_ridge sep2 results_maxmarg sep3 results_edit_alone sep4 results_ensemble_ridge sep5 results_ensemble_maxmarg sep6 results_ensemble_edit_ridge sep7 results_ensemble_edit_maxmarg sep8

sep%:
	@ echo "======================================================================================================="

# in order to keep intermediate files
.SECONDARY:

############################################################################
############################################################################

$(DIR_DATA)/data.zip:
	wget $(URL_DATA) -O $@
	unzip $@ -d $(dir $@)
	rm $@

# to download data
get_data: $(DIR_DATA)/data.zip

$(DIR_TEXTS)/% DIR_EMBEDDINGS/%: $(DIR_DATA)/data.zip

############################################################################
############################################################################

define run_ridge_bwe_mapping
	mkdir -p $(dir $(1))
	$(PYTHON) ${DIR_SCRIPTS_RIDGE}/bwe_mapping.py -d $(2) --from en --to de --po $@ --fv $(3) --tv $(4)
endef

$(DIR_RESULTS_RIDGE)/mappedBWE.General.%.en-de: $(LEXICON_TRAIN_GENERAL) $(DIR_EMBEDDINGS_GENERAL)/embeddings.%.en $(DIR_EMBEDDINGS_GENERAL)/embeddings.%.de
	$(call run_ridge_bwe_mapping,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

$(DIR_RESULTS_RIDGE)/mappedBWE.Medical.%.en-de: $(LEXICON_TRAIN_MEDICAL) $(DIR_EMBEDDINGS_MEDICAL)/embeddings.%.en $(DIR_EMBEDDINGS_MEDICAL)/embeddings.%.de
	$(call run_ridge_bwe_mapping,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

ridge_mappings: $(DIR_RESULTS_RIDGE)/mappedBWE.General.skip.en-de $(DIR_RESULTS_RIDGE)/mappedBWE.General.cbow.en-de $(DIR_RESULTS_RIDGE)/mappedBWE.General.ftskip.en-de $(DIR_RESULTS_RIDGE)/mappedBWE.General.ftcbow.en-de \
			    $(DIR_RESULTS_RIDGE)/mappedBWE.Medical.skip.en-de $(DIR_RESULTS_RIDGE)/mappedBWE.Medical.cbow.en-de $(DIR_RESULTS_RIDGE)/mappedBWE.Medical.ftskip.en-de $(DIR_RESULTS_RIDGE)/mappedBWE.Medical.ftcbow.en-de

############################################################################

define run_ridge_compute_most_sim
	mkdir -p $(dir $(1))tmp
	$(BASH) "$(PYTHON) ${DIR_SCRIPTS_RIDGE}/computeMostSimilar.py -s <(cut -f 1 $(2)) -v $(3) -t $(4) -o $(1) > $(dir $(1))tmp/notInEmbedding.$(notdir $(1)) -n $(5)"
endef

$(DIR_RESULTS_RIDGE)/minedOOV.General.%.txt: $(LEXICON_TEST_GENERAL) $(DIR_RESULTS_RIDGE)/mappedBWE.General.%.en-de $(DIR_EMBEDDINGS_GENERAL)/embeddings.%.de
	$(call run_ridge_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),5)

$(DIR_RESULTS_RIDGE)/minedOOV.Crawl.%.txt: $(LEXICON_TEST_CRAWL) $(DIR_RESULTS_RIDGE)/mappedBWE.General.%.en-de $(DIR_EMBEDDINGS_GENERAL)/embeddings.%.de
	$(call run_ridge_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),5)

$(DIR_RESULTS_RIDGE)/minedOOV.News.%.txt: $(LEXICON_TEST_NEWS) $(DIR_RESULTS_RIDGE)/mappedBWE.General.%.en-de $(DIR_EMBEDDINGS_GENERAL)/embeddings.%.de
	$(call run_ridge_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),5)

$(DIR_RESULTS_RIDGE)/minedOOV.Himl.%.txt:  $(LEXICON_TEST_HIML) $(DIR_RESULTS_RIDGE)/mappedBWE.Medical.%.en-de $(DIR_EMBEDDINGS_MEDICAL)/embeddings.%.de
	$(call run_ridge_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),5)

$(DIR_RESULTS_RIDGE)/minedOOV.RareHiml.%.txt: $(LEXICON_TEST_RAREHIML) $(DIR_RESULTS_RIDGE)/mappedBWE.Medical.%.en-de $(DIR_EMBEDDINGS_MEDICAL)/embeddings.%.de
	$(call run_ridge_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),5)

############################################################################


define run_eval
	mkdir -p $(dir $(1))
	$(PYTHON) ${DIR_SCRIPTS}/checkIfInSeed.py -d $(2) -n $(3) -f $(1).NBEST.txt -s $(1).1BEST.txt
	$(PYTHON) ${DIR_SCRIPTS}/countMatches.py 0 < $(1).1BEST.txt > $(1).1BEST.txt.result
	$(PYTHON) ${DIR_SCRIPTS}/countMatches.py 0 < $(1).NBEST.txt > $(1).NBEST.txt.result
endef

# NBEST is not checked, but this is better for parallel execution
$(DIR_RESULTS_RIDGE)/results.General.%.1BEST.txt.result: $(LEXICON_TEST_GENERAL) $(DIR_RESULTS_RIDGE)/minedOOV.General.%.txt
	$(call run_eval,$(subst .1BEST.txt.result,,$@),$(word 1,$^),$(word 2,$^))

$(DIR_RESULTS_RIDGE)/results.Crawl.%.1BEST.txt.result: $(LEXICON_TEST_CRAWL) $(DIR_RESULTS_RIDGE)/minedOOV.Crawl.%.txt
	$(call run_eval,$(subst .1BEST.txt.result,,$@),$(word 1,$^),$(word 2,$^))

$(DIR_RESULTS_RIDGE)/results.News.%.1BEST.txt.result: $(LEXICON_TEST_NEWS) $(DIR_RESULTS_RIDGE)/minedOOV.News.%.txt
	$(call run_eval,$(subst .1BEST.txt.result,,$@),$(word 1,$^),$(word 2,$^))

$(DIR_RESULTS_RIDGE)/results.Himl.%.1BEST.txt.result: $(LEXICON_TEST_HIML) $(DIR_RESULTS_RIDGE)/minedOOV.Himl.%.txt
	$(call run_eval,$(subst .1BEST.txt.result,,$@),$(word 1,$^),$(word 2,$^))

$(DIR_RESULTS_RIDGE)/results.RareHiml.%.1BEST.txt.result: $(LEXICON_TEST_RAREHIML) $(DIR_RESULTS_RIDGE)/minedOOV.RareHiml.%.txt
	$(call run_eval,$(subst .1BEST.txt.result,,$@),$(word 1,$^),$(word 2,$^))

############################################################################

define print_result
@ echo "$(2): $(subst results.,,$(notdir $(1))): `cut -d ',' -f 1 $(1).1BEST.txt.result` (`cut -d ',' -f 1 $(1).NBEST.txt.result`)"
endef

results_ridge_%: $(DIR_RESULTS_RIDGE)/results.%.1BEST.txt.result
	$(call print_result,$(subst .1BEST.txt.result,,$(word 1,$^)),RIDGE)

results_ridge: results_ridge_General.skip results_ridge_General.cbow results_ridge_General.ftskip results_ridge_General.ftcbow \
			   results_ridge_Himl.skip results_ridge_Himl.cbow results_ridge_Himl.ftskip results_ridge_Himl.ftcbow \
			   results_ridge_Crawl.skip results_ridge_Crawl.cbow results_ridge_Crawl.ftskip results_ridge_Crawl.ftcbow \
			   results_ridge_News.skip results_ridge_News.cbow results_ridge_News.ftskip results_ridge_News.ftcbow \
			   results_ridge_RareHiml.skip results_ridge_RareHiml.cbow results_ridge_RareHiml.ftskip results_ridge_RareHiml.ftcbow


############################################################################
############################################################################

define create_triplets_singlesource
	mkdir -p $(dir $(1))
	$(PYTHON) ${DIR_SCRIPTS_MAXMARG}/createTripletsSingleSource.py $(3) $(2) $(1) $(1).source $(1).target $(4)
endef

$(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.%.txt: $(LEXICON_TRAIN_GENERAL) $(DIR_TEXTS)/general.de
	$(call create_triplets_singlesource,$@,$(word 1,$^),$(word 2,$^),$*)

$(DIR_RESULTS_MAXMARG_TRIPLETS)/medical_triplets_singlesource.%.txt: $(LEXICON_TRAIN_MEDICAL) $(DIR_TEXTS)/medical.de
	$(call create_triplets_singlesource,$@,$(word 1,$^),$(word 2,$^),$*)

############################################################################

define run_maxmarg
	mkdir -p $(dir $(1))
	THEANO_FLAGS='floatX=float32,device=$(DEVICE),lib.cnmem=1' $(PYTHON) ${DIR_SCRIPTS_MAXMARG}/tripletModelConcat.py -t $(2) --source $(2).source --target $(2).target --sv $(3) --tv $(4) --proj $@  --val_split 0.25 --m $(5) --ne 100 --b 1
endef

######## GENERAL - GENERAL ######
METHOD = skip
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_GENERAL_SKIP)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = cbow
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_GENERAL_CBOW)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = ftskip
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_GENERAL_FTSKIP)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = ftcbow
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_GENERAL_FTCBOW)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

######## GENERAL - NEWS ######
METHOD = skip
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_NEWS_SKIP)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = cbow
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_NEWS_CBOW)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = ftskip
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_NEWS_FTSKIP)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = ftcbow
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_NEWS_FTCBOW)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

######## GENERAL - CRAWL ######
METHOD = skip
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_CRAWL_SKIP)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = cbow
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_CRAWL_CBOW)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = ftskip
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_CRAWL_FTSKIP)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = ftcbow
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_CRAWL_FTCBOW)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/general_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))


######### MEDICAL - HIML ##############
METHOD = skip
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_HIML_SKIP)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/medical_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = cbow
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_HIML_CBOW)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/medical_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = ftskip
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_HIML_FTSKIP)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/medical_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = ftcbow
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_HIML_FTCBOW)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/medical_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

######### MEDICAL - RAREHIML ##############
METHOD = skip
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_RAREHIML_SKIP)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/medical_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = cbow
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_RAREHIML_CBOW)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/medical_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = ftskip
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_RAREHIML_FTSKIP)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/medical_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

METHOD = ftcbow
N = $(NUM_NEGATIVES)
M = $(NUM_MARG_RAREHIML_FTCBOW)
$(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).$(N).$(M).projection: $(DIR_RESULTS_MAXMARG_TRIPLETS)/medical_triplets_singlesource.$(N).txt  $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).en $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de
	$(call run_maxmarg,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(call split_dot,$@,4))

############################################################################

define run_maxmarg_compute_most_sim
	mkdir -p $(dir $(1))/tmp
	$(BASH) "$(PYTHON) ${DIR_SCRIPTS_MAXMARG}/computeMostSimilar.py -s <(cut -f 1 $(2)) -t $(3) -v $(4) -o $(1) > $(dir $(1))/tmp/notInEmbedding.$(notdir $(1))"
endef

METHOD = skip
$(DIR_RESULTS_MAXMARG)/minedOOV.General.$(METHOD).%.txt: $(LEXICON_TEST_GENERAL) $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

METHOD = cbow
$(DIR_RESULTS_MAXMARG)/minedOOV.General.$(METHOD).%.txt: $(LEXICON_TEST_GENERAL) $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

METHOD = ftskip
$(DIR_RESULTS_MAXMARG)/minedOOV.General.$(METHOD).%.txt: $(LEXICON_TEST_GENERAL) $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

METHOD = ftcbow
$(DIR_RESULTS_MAXMARG)/minedOOV.General.$(METHOD).%.txt: $(LEXICON_TEST_GENERAL) $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

#########################

METHOD = skip
$(DIR_RESULTS_MAXMARG)/minedOOV.News.$(METHOD).%.txt: $(LEXICON_TEST_NEWS) $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

METHOD = cbow
$(DIR_RESULTS_MAXMARG)/minedOOV.News.$(METHOD).%.txt: $(LEXICON_TEST_NEWS) $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

METHOD = ftskip
$(DIR_RESULTS_MAXMARG)/minedOOV.News.$(METHOD).%.txt: $(LEXICON_TEST_NEWS) $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

METHOD = ftcbow
$(DIR_RESULTS_MAXMARG)/minedOOV.News.$(METHOD).%.txt: $(LEXICON_TEST_NEWS) $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

#########################
 
METHOD = skip
$(DIR_RESULTS_MAXMARG)/minedOOV.Crawl.$(METHOD).%.txt: $(LEXICON_TEST_CRAWL) $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))
 
METHOD = cbow
$(DIR_RESULTS_MAXMARG)/minedOOV.Crawl.$(METHOD).%.txt: $(LEXICON_TEST_CRAWL) $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))
 
METHOD = ftskip
$(DIR_RESULTS_MAXMARG)/minedOOV.Crawl.$(METHOD).%.txt: $(LEXICON_TEST_CRAWL) $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))
 
METHOD = ftcbow
$(DIR_RESULTS_MAXMARG)/minedOOV.Crawl.$(METHOD).%.txt: $(LEXICON_TEST_CRAWL) $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

#########################
 
METHOD = skip
$(DIR_RESULTS_MAXMARG)/minedOOV.Himl.$(METHOD).%.txt: $(LEXICON_TEST_HIML) $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))
 
METHOD = cbow
$(DIR_RESULTS_MAXMARG)/minedOOV.Himl.$(METHOD).%.txt: $(LEXICON_TEST_HIML) $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

METHOD = ftskip
$(DIR_RESULTS_MAXMARG)/minedOOV.Himl.$(METHOD).%.txt: $(LEXICON_TEST_HIML) $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))
 
METHOD = ftcbow
$(DIR_RESULTS_MAXMARG)/minedOOV.Himl.$(METHOD).%.txt: $(LEXICON_TEST_HIML) $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

#########################
 
METHOD = skip
$(DIR_RESULTS_MAXMARG)/minedOOV.RareHiml.$(METHOD).%.txt: $(LEXICON_TEST_RAREHIML) $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))
 
METHOD = cbow
$(DIR_RESULTS_MAXMARG)/minedOOV.RareHiml.$(METHOD).%.txt: $(LEXICON_TEST_RAREHIML) $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))
 
METHOD = ftskip
$(DIR_RESULTS_MAXMARG)/minedOOV.RareHiml.$(METHOD).%.txt: $(LEXICON_TEST_RAREHIML) $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))
 
METHOD = ftcbow
$(DIR_RESULTS_MAXMARG)/minedOOV.RareHiml.$(METHOD).%.txt: $(LEXICON_TEST_RAREHIML) $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de $(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).%.projection
	$(call run_maxmarg_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^))

############################################################################

$(DIR_RESULTS_MAXMARG)/results.General.%.1BEST.txt.result: $(LEXICON_TEST_GENERAL) $(DIR_RESULTS_MAXMARG)/minedOOV.General.%.txt
	$(call run_eval,$(subst .1BEST.txt.result,,$@),$(word 1,$^),$(word 2,$^))

$(DIR_RESULTS_MAXMARG)/results.News.%.1BEST.txt.result: $(LEXICON_TEST_NEWS) $(DIR_RESULTS_MAXMARG)/minedOOV.News.%.txt
	$(call run_eval,$(subst .1BEST.txt.result,,$@),$(word 1,$^),$(word 2,$^))

$(DIR_RESULTS_MAXMARG)/results.Crawl.%.1BEST.txt.result: $(LEXICON_TEST_CRAWL) $(DIR_RESULTS_MAXMARG)/minedOOV.Crawl.%.txt
	$(call run_eval,$(subst .1BEST.txt.result,,$@),$(word 1,$^),$(word 2,$^))

$(DIR_RESULTS_MAXMARG)/results.Himl.%.1BEST.txt.result: $(LEXICON_TEST_HIML) $(DIR_RESULTS_MAXMARG)/minedOOV.Himl.%.txt
	$(call run_eval,$(subst .1BEST.txt.result,,$@),$(word 1,$^),$(word 2,$^))

$(DIR_RESULTS_MAXMARG)/results.RareHiml.%.1BEST.txt.result: $(LEXICON_TEST_RAREHIML) $(DIR_RESULTS_MAXMARG)/minedOOV.RareHiml.%.txt
	$(call run_eval,$(subst .1BEST.txt.result,,$@),$(word 1,$^),$(word 2,$^))

############################################################################

results_maxmarg_%: $(DIR_RESULTS_MAXMARG)/results.%.1BEST.txt.result
	$(call print_result,$(subst .1BEST.txt.result,,$(word 1,$^)),MAXMARG)

results_maxmarg: results_maxmarg_General.skip.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_SKIP) results_maxmarg_General.cbow.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_CBOW) results_maxmarg_General.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_FTSKIP) results_maxmarg_General.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_FTCBOW) \
			  	 results_maxmarg_Himl.skip.$(NUM_NEGATIVES).$(NUM_MARG_HIML_SKIP) results_maxmarg_Himl.cbow.$(NUM_NEGATIVES).$(NUM_MARG_HIML_CBOW) results_maxmarg_Himl.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_HIML_FTSKIP) results_maxmarg_Himl.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_HIML_FTCBOW) \
			  	 results_maxmarg_Crawl.skip.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_SKIP) results_maxmarg_Crawl.cbow.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_CBOW) results_maxmarg_Crawl.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_FTSKIP) results_maxmarg_Crawl.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_FTCBOW) \
			  	 results_maxmarg_News.skip.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_SKIP) results_maxmarg_News.cbow.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_CBOW) results_maxmarg_News.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_FTSKIP) results_maxmarg_News.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_FTCBOW) \
			   	 results_maxmarg_RareHiml.skip.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_SKIP) results_maxmarg_RareHiml.cbow.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_CBOW) results_maxmarg_RareHiml.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_FTSKIP) results_maxmarg_RareHiml.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_FTCBOW)

############################################################################
############################################################################

define rule_ensemble_maxmarg_general
$(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.$(METHOD).%.txt: $(LEXICON_TEST_GENERAL) $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$$(call run_ridge_compute_most_sim,$$@,$$(word 1,$$^),$$(word 2,$$^),$$(word 3,$$^),100)
endef

$(foreach METHOD,$(METHODS),$(eval $(rule_ensemble_maxmarg_general)))

define rule_ensemble_maxmarg_news
$(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.$(METHOD).%.txt: $(LEXICON_TEST_NEWS) $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$$(call run_ridge_compute_most_sim,$$@,$$(word 1,$$^),$$(word 2,$$^),$$(word 3,$$^),100)
endef

$(foreach METHOD,$(METHODS),$(eval $(rule_ensemble_maxmarg_news)))

define rule_ensemble_maxmarg_crawl
$(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.$(METHOD).%.txt: $(LEXICON_TEST_CRAWL) $(DIR_RESULTS_MAXMARG_PROJECTIONS)/general.$(METHOD).%.projection $(DIR_EMBEDDINGS_GENERAL)/embeddings.$(METHOD).de
	$$(call run_ridge_compute_most_sim,$$@,$$(word 1,$$^),$$(word 2,$$^),$$(word 3,$$^),100)
endef

$(foreach METHOD,$(METHODS),$(eval $(rule_ensemble_maxmarg_crawl)))

define rule_ensemble_maxmarg_himl
$(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.$(METHOD).%.txt: $(LEXICON_TEST_HIML) $(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).%.projection $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de
	$$(call run_ridge_compute_most_sim,$$@,$$(word 1,$$^),$$(word 2,$$^),$$(word 3,$$^),100)
endef

$(foreach METHOD,$(METHODS),$(eval $(rule_ensemble_maxmarg_himl)))

define rule_ensemble_maxmarg_rarehiml
$(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.$(METHOD).%.txt: $(LEXICON_TEST_RAREHIML) $(DIR_RESULTS_MAXMARG_PROJECTIONS)/medical.$(METHOD).%.projection $(DIR_EMBEDDINGS_MEDICAL)/embeddings.$(METHOD).de
	$$(call run_ridge_compute_most_sim,$$@,$$(word 1,$$^),$$(word 2,$$^),$$(word 3,$$^),100)
endef

$(foreach METHOD,$(METHODS),$(eval $(rule_ensemble_maxmarg_rarehiml)))

############################################################################

define run_levenstein_eval
	mkdir -p $(dir $(1))
	$(PYTHON) ${DIR_SCRIPTS}/average_for_eval_levensthein.py -d $(3) -d $(4) -d $(5) -d $(6) -v $(2) -p $(7)  -p $(8)  -p $(9)  -p $(10) -e $(11) > $(1)
endef

$(DIR_RESULTS_ENSEMBLE)/General_edit_alone.result: $(LEXICON_TEST_GENERAL) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.skip.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.cbow.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_FTCBOW).txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),0.0,0.0,0.0,0.0,1.0)

$(DIR_RESULTS_ENSEMBLE)/News_edit_alone.result: $(LEXICON_TEST_NEWS) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.skip.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.cbow.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_FTCBOW).txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),0.0,0.0,0.0,0.0,1.0)

$(DIR_RESULTS_ENSEMBLE)/Crawl_edit_alone.result: $(LEXICON_TEST_CRAWL) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.skip.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.cbow.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_FTCBOW).txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),0.0,0.0,0.0,0.0,1.0)

$(DIR_RESULTS_ENSEMBLE)/Himl_edit_alone.result: $(LEXICON_TEST_HIML) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.skip.$(NUM_NEGATIVES).$(NUM_MARG_HIML_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.cbow.$(NUM_NEGATIVES).$(NUM_MARG_HIML_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_HIML_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_HIML_FTCBOW).txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),0.0,0.0,0.0,0.0,1.0)

$(DIR_RESULTS_ENSEMBLE)/RareHiml_edit_alone.result: $(LEXICON_TEST_RAREHIML) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.skip.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.cbow.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_FTCBOW).txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),0.0,0.0,0.0,0.0,1.0)

############################################################################

define print_result_ensemble
	@ echo -n "$(5): "
	@ echo -n "$(word 1,$(subst _, ,$(notdir $(1)))) "
	@ tail -n 1 $(1) | $(PYTHON) -c "data = raw_input().split(','); print '{} ({})'.format(float(data[$(2)])/float(data[$(4)]), float(data[$(3)])/float(data[$(4)]))"
endef

results_edit_alone_%: $(DIR_RESULTS_ENSEMBLE)/%_edit_alone.result
	$(call print_result_ensemble,$(word 1,$^),5,6,7,EDIT_ALONE)

results_edit_alone: results_edit_alone_General results_edit_alone_Himl results_edit_alone_Crawl results_edit_alone_News results_edit_alone_RareHiml

############################################################################
######################## ENSEMBLE_RIDGE ####################################

$(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.General.%.txt: $(LEXICON_TEST_GENERAL) $(DIR_RESULTS_RIDGE)/mappedBWE.General.%.en-de $(DIR_EMBEDDINGS_GENERAL)/embeddings.%.de
	$(call run_ridge_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),100)

$(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Crawl.%.txt: $(LEXICON_TEST_CRAWL) $(DIR_RESULTS_RIDGE)/mappedBWE.General.%.en-de $(DIR_EMBEDDINGS_GENERAL)/embeddings.%.de
	$(call run_ridge_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),100)

$(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.News.%.txt: $(LEXICON_TEST_NEWS) $(DIR_RESULTS_RIDGE)/mappedBWE.General.%.en-de $(DIR_EMBEDDINGS_GENERAL)/embeddings.%.de
	$(call run_ridge_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),100)

$(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Himl.%.txt: $(LEXICON_TEST_HIML) $(DIR_RESULTS_RIDGE)/mappedBWE.Medical.%.en-de $(DIR_EMBEDDINGS_MEDICAL)/embeddings.%.de
	$(call run_ridge_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),100)

$(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.RareHiml.%.txt: $(LEXICON_TEST_RAREHIML) $(DIR_RESULTS_RIDGE)/mappedBWE.Medical.%.en-de $(DIR_EMBEDDINGS_MEDICAL)/embeddings.%.de
	$(call run_ridge_compute_most_sim,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),100)

############################################################################

define run_ensemble_eval
	mkdir -p $(dir $(1))
	$(PYTHON) ${DIR_SCRIPTS}/average_for_eval.py -d $(3) -d $(4) -d $(5) -d $(6) -v $(2) -p $(7)  -p $(8)  -p $(9)  -p $(10)  > $(1)
endef

$(DIR_RESULTS_ENSEMBLE_RIDGE)/General_ensemble.%.result: $(LEXICON_TEST_GENERAL) $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.General.skip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.General.cbow.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.General.ftskip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.General.ftcbow.txt
	$(call run_ensemble_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_))

$(DIR_RESULTS_ENSEMBLE_RIDGE)/Crawl_ensemble.%.result: $(LEXICON_TEST_CRAWL) $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Crawl.skip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Crawl.cbow.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Crawl.ftskip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Crawl.ftcbow.txt
	$(call run_ensemble_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_))

$(DIR_RESULTS_ENSEMBLE_RIDGE)/News_ensemble.%.result: $(LEXICON_TEST_NEWS) $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.News.skip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.News.cbow.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.News.ftskip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.News.ftcbow.txt
	$(call run_ensemble_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_))

$(DIR_RESULTS_ENSEMBLE_RIDGE)/Himl_ensemble.%.result: $(LEXICON_TEST_HIML) $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Himl.skip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Himl.cbow.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Himl.ftskip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Himl.ftcbow.txt
	$(call run_ensemble_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_))

$(DIR_RESULTS_ENSEMBLE_RIDGE)/RareHiml_ensemble.%.result: $(LEXICON_TEST_RAREHIML) $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.RareHiml.skip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.RareHiml.cbow.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.RareHiml.ftskip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.RareHiml.ftcbow.txt
	$(call run_ensemble_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_))

############################################################################

results_ensemble_ridge_General: $(DIR_RESULTS_ENSEMBLE_RIDGE)/General_ensemble.$(WEIGHTS_ENSEMBLE_RIDGE_GENERAL).result
	$(call print_result_ensemble,$(word 1,$^),0,1,2,ENSEMBLE_RIDGE)

results_ensemble_ridge_Crawl: $(DIR_RESULTS_ENSEMBLE_RIDGE)/Crawl_ensemble.$(WEIGHTS_ENSEMBLE_RIDGE_CRAWL).result
	$(call print_result_ensemble,$(word 1,$^),0,1,2,ENSEMBLE_RIDGE)

results_ensemble_ridge_News: $(DIR_RESULTS_ENSEMBLE_RIDGE)/News_ensemble.$(WEIGHTS_ENSEMBLE_RIDGE_NEWS).result
	$(call print_result_ensemble,$(word 1,$^),0,1,2,ENSEMBLE_RIDGE)

results_ensemble_ridge_Himl: $(DIR_RESULTS_ENSEMBLE_RIDGE)/Himl_ensemble.$(WEIGHTS_ENSEMBLE_RIDGE_HIML).result
	$(call print_result_ensemble,$(word 1,$^),0,1,2,ENSEMBLE_RIDGE)

results_ensemble_ridge_RareHiml: $(DIR_RESULTS_ENSEMBLE_RIDGE)/RareHiml_ensemble.$(WEIGHTS_ENSEMBLE_RIDGE_RAREHIML).result
	$(call print_result_ensemble,$(word 1,$^),0,1,2,ENSEMBLE_RIDGE)


results_ensemble_ridge: results_ensemble_ridge_General results_ensemble_ridge_Himl results_ensemble_ridge_Crawl results_ensemble_ridge_News results_ensemble_ridge_RareHiml

############################################################################
####################### ENSEMBLE_MAXMARG ###################################

$(DIR_RESULTS_ENSEMBLE_MAXMARG)/General_ensemble.%.result: $(LEXICON_TEST_GENERAL) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.skip.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.cbow.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_FTCBOW).txt
	$(call run_ensemble_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_))

$(DIR_RESULTS_ENSEMBLE_MAXMARG)/News_ensemble.%.result: $(LEXICON_TEST_NEWS) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.skip.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.cbow.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_FTCBOW).txt
	$(call run_ensemble_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_))

$(DIR_RESULTS_ENSEMBLE_MAXMARG)/Crawl_ensemble.%.result: $(LEXICON_TEST_CRAWL) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.skip.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.cbow.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_FTCBOW).txt
	$(call run_ensemble_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_))

$(DIR_RESULTS_ENSEMBLE_MAXMARG)/Himl_ensemble.%.result: $(LEXICON_TEST_HIML) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.skip.$(NUM_NEGATIVES).$(NUM_MARG_HIML_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.cbow.$(NUM_NEGATIVES).$(NUM_MARG_HIML_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_HIML_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_HIML_FTCBOW).txt
	$(call run_ensemble_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_))

$(DIR_RESULTS_ENSEMBLE_MAXMARG)/RareHiml_ensemble.%.result: $(LEXICON_TEST_RAREHIML) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.skip.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.cbow.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_FTCBOW).txt
	$(call run_ensemble_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_))

############################################################################

results_ensemble_maxmarg_General: $(DIR_RESULTS_ENSEMBLE_MAXMARG)/General_ensemble.$(WEIGHTS_ENSEMBLE_MAXMARG_GENERAL).result
	$(call print_result_ensemble,$(word 1,$^),0,1,2,ENSEMBLE_MAXMARG)

results_ensemble_maxmarg_Crawl: $(DIR_RESULTS_ENSEMBLE_MAXMARG)/Crawl_ensemble.$(WEIGHTS_ENSEMBLE_MAXMARG_CRAWL).result
	$(call print_result_ensemble,$(word 1,$^),0,1,2,ENSEMBLE_MAXMARG)

results_ensemble_maxmarg_News: $(DIR_RESULTS_ENSEMBLE_MAXMARG)/News_ensemble.$(WEIGHTS_ENSEMBLE_MAXMARG_NEWS).result
	$(call print_result_ensemble,$(word 1,$^),0,1,2,ENSEMBLE_MAXMARG)

results_ensemble_maxmarg_Himl: $(DIR_RESULTS_ENSEMBLE_MAXMARG)/Himl_ensemble.$(WEIGHTS_ENSEMBLE_MAXMARG_HIML).result
	$(call print_result_ensemble,$(word 1,$^),0,1,2,ENSEMBLE_MAXMARG)

results_ensemble_maxmarg_RareHiml: $(DIR_RESULTS_ENSEMBLE_MAXMARG)/RareHiml_ensemble.$(WEIGHTS_ENSEMBLE_MAXMARG_RAREHIML).result
	$(call print_result_ensemble,$(word 1,$^),0,1,2,ENSEMBLE_MAXMARG)


results_ensemble_maxmarg: results_ensemble_maxmarg_General results_ensemble_maxmarg_Himl results_ensemble_maxmarg_Crawl results_ensemble_maxmarg_News results_ensemble_maxmarg_RareHiml

############################################################################
######################## ENSEMBLE_EDIT_RIDGE ###############################

$(DIR_RESULTS_ENSEMBLE_RIDGE)/General_ensemble_edit.%.result: $(LEXICON_TEST_GENERAL) $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.General.skip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.General.cbow.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.General.ftskip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.General.ftcbow.txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_),$(call split,$*,5,_))

$(DIR_RESULTS_ENSEMBLE_RIDGE)/Crawl_ensemble_edit.%.result: $(LEXICON_TEST_CRAWL) $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Crawl.skip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Crawl.cbow.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Crawl.ftskip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Crawl.ftcbow.txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_),$(call split,$*,5,_))

$(DIR_RESULTS_ENSEMBLE_RIDGE)/News_ensemble_edit.%.result: $(LEXICON_TEST_NEWS) $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.News.skip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.News.cbow.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.News.ftskip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.News.ftcbow.txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_),$(call split,$*,5,_))

$(DIR_RESULTS_ENSEMBLE_RIDGE)/Himl_ensemble_edit.%.result: $(LEXICON_TEST_HIML) $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Himl.skip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Himl.cbow.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Himl.ftskip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.Himl.ftcbow.txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_),$(call split,$*,5,_))

$(DIR_RESULTS_ENSEMBLE_RIDGE)/RareHiml_ensemble_edit.%.result: $(LEXICON_TEST_RAREHIML) $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.RareHiml.skip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.RareHiml.cbow.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.RareHiml.ftskip.txt $(DIR_RESULTS_ENSEMBLE_RIDGE)/minedOOV.RareHiml.ftcbow.txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_),$(call split,$*,5,_))

############################################################################

results_ensemble_edit_ridge_General: $(DIR_RESULTS_ENSEMBLE_RIDGE)/General_ensemble_edit.$(WEIGHTS_ENSEMBLE_EDIT_RIDGE_GENERAL).result
	$(call print_result_ensemble,$(word 1,$^),5,6,7,ENSEMBLE_EDIT_RIDGE)

results_ensemble_edit_ridge_Crawl: $(DIR_RESULTS_ENSEMBLE_RIDGE)/Crawl_ensemble_edit.$(WEIGHTS_ENSEMBLE_EDIT_RIDGE_CRAWL).result
	$(call print_result_ensemble,$(word 1,$^),5,6,7,ENSEMBLE_EDIT_RIDGE)

results_ensemble_edit_ridge_News: $(DIR_RESULTS_ENSEMBLE_RIDGE)/News_ensemble_edit.$(WEIGHTS_ENSEMBLE_EDIT_RIDGE_NEWS).result
	$(call print_result_ensemble,$(word 1,$^),5,6,7,ENSEMBLE_EDIT_RIDGE)

results_ensemble_edit_ridge_Himl: $(DIR_RESULTS_ENSEMBLE_RIDGE)/Himl_ensemble_edit.$(WEIGHTS_ENSEMBLE_EDIT_RIDGE_HIML).result
	$(call print_result_ensemble,$(word 1,$^),5,6,7,ENSEMBLE_EDIT_RIDGE)

results_ensemble_edit_ridge_RareHiml: $(DIR_RESULTS_ENSEMBLE_RIDGE)/RareHiml_ensemble_edit.$(WEIGHTS_ENSEMBLE_EDIT_RIDGE_RAREHIML).result
	$(call print_result_ensemble,$(word 1,$^),5,6,7,ENSEMBLE_EDIT_RIDGE)


results_ensemble_edit_ridge: results_ensemble_edit_ridge_General results_ensemble_edit_ridge_Himl results_ensemble_edit_ridge_Crawl results_ensemble_edit_ridge_News results_ensemble_edit_ridge_RareHiml

############################################################################
######################## ENSEMBLE_EDIT_MAXMARG #############################

$(DIR_RESULTS_ENSEMBLE_MAXMARG)/General_ensemble_edit.%.result: $(LEXICON_TEST_GENERAL) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.skip.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.cbow.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.General.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_GENERAL_FTCBOW).txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_),$(call split,$*,5,_))

$(DIR_RESULTS_ENSEMBLE_MAXMARG)/News_ensemble_edit.%.result: $(LEXICON_TEST_NEWS) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.skip.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.cbow.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.News.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_NEWS_FTCBOW).txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_),$(call split,$*,5,_))

$(DIR_RESULTS_ENSEMBLE_MAXMARG)/Crawl_ensemble_edit.%.result: $(LEXICON_TEST_CRAWL) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.skip.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.cbow.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Crawl.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_CRAWL_FTCBOW).txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_),$(call split,$*,5,_))

$(DIR_RESULTS_ENSEMBLE_MAXMARG)/Himl_ensemble_edit.%.result: $(LEXICON_TEST_HIML) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.skip.$(NUM_NEGATIVES).$(NUM_MARG_HIML_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.cbow.$(NUM_NEGATIVES).$(NUM_MARG_HIML_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_HIML_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.Himl.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_HIML_FTCBOW).txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_),$(call split,$*,5,_))

$(DIR_RESULTS_ENSEMBLE_MAXMARG)/RareHiml_ensemble_edit.%.result: $(LEXICON_TEST_RAREHIML) $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.skip.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_SKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.cbow.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_CBOW).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.ftskip.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_FTSKIP).txt $(DIR_RESULTS_ENSEMBLE_MAXMARG)/minedOOV.RareHiml.ftcbow.$(NUM_NEGATIVES).$(NUM_MARG_RAREHIML_FTCBOW).txt
	$(call run_levenstein_eval,$@,$(word 1,$^),$(word 2,$^),$(word 3,$^),$(word 4,$^),$(word 5,$^),$(call split,$*,1,_),$(call split,$*,2,_),$(call split,$*,3,_),$(call split,$*,4,_),$(call split,$*,5,_))

############################################################################

results_ensemble_edit_maxmarg_General: $(DIR_RESULTS_ENSEMBLE_MAXMARG)/General_ensemble_edit.$(WEIGHTS_ENSEMBLE_EDIT_MAXMARG_GENERAL).result
	$(call print_result_ensemble,$(word 1,$^),5,6,7,ENSEMBLE_EDIT_MAXMARG)

results_ensemble_edit_maxmarg_Crawl: $(DIR_RESULTS_ENSEMBLE_MAXMARG)/Crawl_ensemble_edit.$(WEIGHTS_ENSEMBLE_EDIT_MAXMARG_CRAWL).result
	$(call print_result_ensemble,$(word 1,$^),5,6,7,ENSEMBLE_EDIT_MAXMARG)

results_ensemble_edit_maxmarg_News: $(DIR_RESULTS_ENSEMBLE_MAXMARG)/News_ensemble_edit.$(WEIGHTS_ENSEMBLE_EDIT_MAXMARG_NEWS).result
	$(call print_result_ensemble,$(word 1,$^),5,6,7,ENSEMBLE_EDIT_MAXMARG)

results_ensemble_edit_maxmarg_Himl: $(DIR_RESULTS_ENSEMBLE_MAXMARG)/Himl_ensemble_edit.$(WEIGHTS_ENSEMBLE_EDIT_MAXMARG_HIML).result
	$(call print_result_ensemble,$(word 1,$^),5,6,7,ENSEMBLE_EDIT_MAXMARG)

results_ensemble_edit_maxmarg_RareHiml: $(DIR_RESULTS_ENSEMBLE_MAXMARG)/RareHiml_ensemble_edit.$(WEIGHTS_ENSEMBLE_EDIT_MAXMARG_RAREHIML).result
	$(call print_result_ensemble,$(word 1,$^),5,6,7,ENSEMBLE_EDIT_MAXMARG)


results_ensemble_edit_maxmarg: results_ensemble_edit_maxmarg_General results_ensemble_edit_maxmarg_Himl results_ensemble_edit_maxmarg_Crawl results_ensemble_edit_maxmarg_News results_ensemble_edit_maxmarg_RareHiml

