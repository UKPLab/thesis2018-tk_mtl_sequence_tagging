#!/bin/bash
# Download CoNLL 2000 chunking data from https://github.com/teropa/nlp/tree/master/resources/corpora/conll2000

SCRIPT_DIR=`dirname $BASH_SOURCE`

mkdir -p $SCRIPT_DIR/data
cd $SCRIPT_DIR/data
wget https://raw.githubusercontent.com/teropa/nlp/master/resources/corpora/conll2000/train.txt
wget https://raw.githubusercontent.com/teropa/nlp/master/resources/corpora/conll2000/test.txt