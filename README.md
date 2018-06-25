# TensorFlow MTL Sequence Tagging Framework

This MTL sequence tagging framework allows to train any task that can be framed
as a sequence tagging problem, i.e., POS tagging, Chunking, NER, token-level Argumentation Mining, Grapheme-to-Phoneme Conversion, Lemmatization, etc. 

Moreover, it allows to do multi-task learning
to improve the performance compared to single-task learning approaches.

This framework supports CRF output layers, learned character-level word representations (besides standard word embedding based word representations), and different tasks feeding from different layers in the neural network.
The model implements hard parameter-sharing. Our model is an extension of the model of Sogaard & Levy described in [here](http://anthology.aclweb.org/P16-2038). Our model is similar to the following [Keras implementation](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf). See also the references given there.
This TensorFlow implementation offers high and easy configurability.


It was used a.o. in the following papers:

1. Eger, Steffen ; Daxenberger, Johannes ; Stab, Christian ; Gurevych, Iryna :
Cross-lingual Argumentation Mining: Machine Translation (and a bit of Projection) is All You Need!
2. Do Dinh, Erik-Lân ; Eger, Steffen ; Gurevych, Iryna :
Killing four birds with two stones: Multi-Task Learning for Non-Literal Language Detection.

## Using this software
> **NOTE**
> All relative paths mentioned in this README, e.g. `cd src`, refer to the
> root directory of this repository, i.e. the directory which also contains
> this `README.md` file.


Please cite the thesis when using this software:

```
@mastersthesis{Kahse:2018,
  author       = {Tobias Kahse}, 
  title        = {Multi-Task Learning For Argumentation Mining},
  school       = {Technical University Darmstadt},
  year         = 2018,
}
```

If you have further questions, contact Tobias Kahse (tobias.kahse@outlook.com). Further contact persons are Steffen Eger and Johannes Daxenberger.

## Requirements

To use the framework, Python 2.7.x is required. All other dependencies are listed
in [requirements.txt](./requirements.txt).

You can either use the framework within your global environment or create a virtual
environment by running

```bash
virtualenv .
```

within this directory. However, this requires you to install `virtualenv` (see [here](https://virtualenv.pypa.io/en/stable/installation/)).

If you are using a virtual environment, make sure to [activate](https://virtualenv.pypa.io/en/stable/userguide/#activate-script) it before installing the requirements or executing any python script. The requirements can be installed by running

```bash
pip install -r requirements.txt
```

## Configuration

To ensure flexibility of the framework, (almost) everything can be configured
within a YAML file. See [CONFIGURATION.md](./CONFIGURATION.md) for information
about the contents of the configuration file. You should use a Markdown viewer
(almost any editor or IDE should have one or a plugin) to view this file.

## Training

To train a model, you have to write a configuration file. If you have a configuration
file, just run the command

```bash
cd src
python main.py train PATH_TO_YOUR_CONFIG
```

## Evaluate

Given a trained and saved model, an evaluation on the test dataset can be performed as follows

```bash
cd src
python main.py eval PATH_TO_YOUR_SAVED_MODEL PATH_TO_YOUR_CONFIG
```

The evaluation is performed on the test dataset which is specified in the configuration file.

## Evaluate Session
Whenever a configuration file is used to train, the `num_runs` configuration option specifies how many models are created. All these
runs are associated with a "session". To evaluate all models of a
session on test at the same time, use the following command

```bash
cd src
python main.py eval-session PATH_TO_YOUR_SESSION OUT_PATH PATH_TO_YOUR_CONFIG
```

The parameters are as follows:
* `PATH_TO_YOUR_SESSION`: Path to the session folder with the different runs.
* `OUT_PATH`: Path to the output folder where the prediction files of this evaluation are written to.
* `PATH_TO_YOUR_CONFIG`: Path to the configuration in which we specify the test file.

## Predict
`TBD`

## Hyper-Parameter Optimization

To use the hyper-parameter optimization, a special configuration file format with two YAML documents is necessary.
See [CONFIGURATION.md](./CONFIGURATION.md) for this format. Then, the following command starts the hyper-parameter
search

```bash
cd src
python run_experiment NUM_TRIALS PATH_TO_YOUR_CONFIG OUTPUT_FOLDER_FOR_CONFIG_FILES
```

The parameters are as follows:

* `NUM_TRIALS`: How many trials of the random search shall be performed. Each new trail tests a new, randomly sampled configuration.
* `PATH_TO_YOUR_CONFIG`: Path to the hyper-parameter search configuration file which is used to create the trails.
* `OUTPUT_FOLDER_FOR_CONFIG_FILES`: Where to store the randomly sampled configuration files.
