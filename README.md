# Unsupervised Scalable Representation Learning for Multivariate Time Series -- Code

This is the code corresponding to the experiments conducted for the work "Unsupervised Scalable Representation Learning for Multivariate Time Series" (Jean-Yves Franceschi, Aymeric Dieuleveut and Martin Jaggi) [[NeurIPS]](https://papers.nips.cc/paper/8713-unsupervised-scalable-representation-learning-for-multivariate-time-series) [[arXiv]](https://arxiv.org/abs/1901.10738) [[HAL]](https://hal.archives-ouvertes.fr/hal-01998101), presented at NeurIPS 2019.
A previous version was presented at the [2nd LLD workshop](https://lld-workshop.github.io/) at ICLR 2019.

## Requirements

Experiments were done with the following package versions for Python 3.6:
 - Numpy (`numpy`) v1.15.2;
 - Matplotlib (`matplotlib`) v3.0.0;
 - Orange (`Orange`) v3.18.0;
 - Pandas (`pandas`) v0.23.4;
 - `python-weka-wrapper3` v0.1.6 for multivariate time series (requires Oracle JDK 8 or OpenJDK 8);
 - PyTorch (`torch`) v0.4.1 with CUDA 9.0;
 - Scikit-learn (`sklearn`) v0.20.0;
 - Scipy (`scipy`) v1.1.0.

This code should execute correctly with updated versions of these packages.

## Datasets

The datasets manipulated in this code can be downloaded on the following locations:
 - the UCR archive: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/;
 - the UEA archive: http://www.timeseriesclassification.com/;
 - the Individual Household Electric Power Consumption dataset:
   https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption.

## Files

### Core

 - `losses` folder: implements the triplet loss in the cases of a training set
   with all time series of the same length, and a training set with time series
   of unequal lengths;
 - `networks` folder: implements encoder and its building blocks (dilated
   convolutions, causal CNN);
 - `scikit_wrappers.py` file: implements classes inheriting Scikit-learn
   classifiers that wrap an encoder and a SVM classifier.
 - `utils.py` file: implements custom PyTorch datasets;
 - `default_hyperparameters.json` file: example of a JSON file containing the
   hyperparameters of a pair (encoder, classifier).

### Tests

 - `ucr.py` file: handles learning on the UCR archive (see usage below);
 - `uea.py` file: handles learning on the UEA archive (see usage below);
 - `transfer_ucr.py` file: handles transfer learning on the UCR archive (see
   usage below);
 - `combine_ucr.py` file: combines learned pairs of (encoder, classifier) for
   the UCR archive) (see usage below);
 - `combine_uea.py` file: combines learned pairs of (encoder, classifier) for
   the UEA archive) (see usage below);
 - `sparse_labeling.ipynb` file: file containing code to reproduce the results
   of training an SVM on our representations for different numbers of available
   labels;
 - `HouseholdPowerConsumption.ipynb` file: Jupyter notebook containing
   experiments on the Individual Household Electric Power Consumption dataset.

### Results and Visualization

 - `results_ucr.csv` file: CSV file compiling all results (with those of
   concurrent methods) on the UCR archive;
 - `results_uea.csv` file: CSV file compiling all results (with those of
   concurrent methods) on the UEA archive;
 - `results_sparse_labeling_TwoPatterns.csv` file: CSV file compiling means and
   standard variations of five runs of learning an SVM on our representations
   and the ResNet architecture described in the paper for different numbers
   of available labels;
 - `cd.ipynb` file: Jupyter notebook containing the code to produce a critical
   difference diagram;
 - `stat_plots.ipynb` file: Jupyter notebook containing the code to produce
   boxplots and histograms on the results for the UCR archive;
 - `models` folder: contains a pretrained model for the UCR dataset CricketX.

## Usage

### Training on the UCR and UEA archives

To train a model on the Mallat dataset from the UCR archive:

`python3 ucr.py --dataset Mallat --path path/to/Mallat/folder/ --save_path /path/to/save/models --hyper default_hyperparameters.json [--cuda --gpu 0]`

Adding the `--load` option allows to load a model from the specified save path.
Training on the UEA archive with `uea.py` is done in a similar way.

### Further Documentation

See the code documentation for more details. `ucr.py`, `uea.py`,
`transfer_ucr.py`, `combine_ucr.py` and `combine_uea.py` can be called with the
`-h` option for additional help.

### Hyperparameters

Hyperparameters are described in Section S2.2 of the paper.

For the UCR and UEA hyperparameters, two values were switched by mistake.
One should read, as reflected in [the example configuration file](default_hyperparameters.json):
> - number of output channels of the causal network (before max pooling): 160;
> - dimension of the representations: 320.
>
instead of
> - number of output channels of the causal network (before max pooling): 320;
> - dimension of the representations: 160.

## Pretrained Models

Pretrained models are downloadable at [https://data.lip6.fr/usrlts/](https://data.lip6.fr/usrlts/).
