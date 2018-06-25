# Grapheme to Phoneme Example

## Introduction

This example presents a configuration to solve a grapheme to phoneme (G2P) task,
i.e. given the letters of a word, we want to know how to pronounce the word.
As an auxiliary task, we choose syllabification of words. The intuition behind
this approach is as follows:

> Given the word "grasshopper" we could pronounce the two letters "sh" either
> as "s" and "h" (which is the correct pronunciation, of course) or as a single
> phoneme "sh" as in "fish".
> It is easier to decide which pronunciation is correct if we know the
> syllables of this word: "grass-hop-per". With this knowledge, only one
> pronunciation is possible.

Since graphemes and phonemes are not necessarily of the same length for one word,
they need to be aligned so that the problem of G2P conversion can be solved
by a sequence tagger. The data in [./data/celex](./data/celex) has been aligned
with the [m2m-aligner](https://github.com/letter-to-phoneme/m2m-aligner) by 

```
@InProceedings{jiampojamarn2007:,
  author    = {Jiampojamarn, Sittichai  and  Kondrak, Grzegorz  and  Sherif, Tarek},
  title     = {Applying Many-to-Many Alignments and Hidden Markov Models to Letter-to-Phoneme Conversion},
  booktitle = {Human Language Technologies 2007: The Conference of the North American Chapter of the Association for Computational Linguistics; Proceedings of the Main Conference},
  month     = {April},
  year      = {2007},
  address   = {Rochester, New York},
  publisher = {Association for Computational Linguistics},
  pages     = {372--379},
  url       = {http://www.aclweb.org/anthology/N/N07/N07-1047}
}
```

The syllabification task is modeled as follows: the network has to map a word to a binary
representation of its syllable splits. The end of each syllable is indicated by a 1. All other
letters are mapped to 0s. Example:

```
    g r a s s h o p p e r
    0 0 0 0 1 0 0 1 0 0 1
```

## Training

Two configurations are provided:

* [stl-configuration-fixed.yaml](./stl-configuration-fixed.yaml) for single task learning (STL)
* [mtl-configuration-fixed.yaml](./mtl-configuration-fixed.yaml) for multi task learning (MTL)

You can run these configurations as described in the "Training" section in the [main README file](../../README.md).

## Results
The results for the Celex G2P task are averaged over two runs. 

### STL
* Accuracy: 58.1%
* Average edit distance: 3.4

### MTL
* Accuracy: 60.1%
* Average edit distance: 3.2
