# CoNLL 2000 Chunking Example
This example presents chunking as in the CoNLL 2000 shared task.
Use `download.sh` to download the data files.

* [configuration-fixed.yaml](./configuration-fixed.yaml): Configuration with fixed parameters.
* [configuration-hyper-param-search.yaml](./configuration-hyper-param-search.yaml): Configuration for a hyper-parameter search.

Running the configuration with fixed parameters results in

* POS tagging accuracy: 96.86%
* Chunking F1: 67.30%

Both scores are averaged across two runs.