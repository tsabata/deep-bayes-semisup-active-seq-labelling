# Deep Bayesian Semi-Supervised Active Learning for Sequence Labelling

This repository contains experiments for article: Deep Bayesian Semi-Supervised Active Learning for Sequence Labelling.

## Getting started

1. Install environment and activate it

    `conda env update & source activate deep-bayes-active-semisup`

2. Download [Glove embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip) and store it in:

`datasets/glove/glove.840B.300d.txt`

3. Run experiment

* BI-LSTM-FCN - experiment with unlimited number of tokens

```   
    python -u experiments/bilstm_fcn_unlimited  NER    
    python -u experiments/bilstm_fcn_unlimited  POS
    python -u experiments/bilstm_fcn_unlimited  CHUNK
 ```

* BI-LSTM-FCN - experiment with limited number of tokens

```   
    python -u experiments/bilstm_fcn_limited  NER    
    python -u experiments/bilstm_fcn_limited  POS
    python -u experiments/bilstm_fcn_limited  CHUNK
 ```
 
* BI-LSTM-CRF - experiment with limited number of tokens

```   
    python -u experiments/bilstm_crf_limited  NER    
    python -u experiments/bilstm_crf_limited  POS
    python -u experiments/bilstm_crf_limited  CHUNK
 ```
 
* BI-LSTM-CRF - experiment with limited number of tokens

```   
    python -u experiments/bilstm_crf_limited  NER    
    python -u experiments/bilstm_crf_limited  POS
    python -u experiments/bilstm_crf_limited  CHUNK
 ```
 