# On the Perils of Cascading Robust Classifiers
The repository supports the paper "On the Perils of Cascading Robust Classifiers."
This repository evaluates the ensemble results under different strategies for the Kolter-Wong models.
See more about the Kolter-Wong models here: https://github.com/locuslab/convex_adversarial/tree/2cd8149249b9e90383af10fc7e9b9fe90166813e

## What is this repository used for?
Cascading ensembles are a popular instance of black-box ensembles that appear to improve certified robust accuracies in practice. 
However, we find that the robustness certifier used by a cascading ensemble is unsound. 
The experimental support for cascading ensemble (unsound) and alternatively weighted voting ensemble (sound) is here. 
For all ensemble strategies, we used pre-trained and self-trained Kolter-Wong models.

## Visualization of Cascading Ensemble unsoundness

<p align = "center">
<img src = "examples/ensemble.png">
</p>
<p align = "left">
These figures are visualizing classification results of 2D points for constituent models (a-c) and the corresponding Cascading Ensemble (d) and Uniform Voting Ensemble (e). Regions with colors correspond to predictions (0: red, 1: blue, 2: green) made by the underlying model (or ensemble). Darker colors indicate that the accompanying robustness certification of the underlying model (or ensemble) returns 1 and lighter colors are for cases when the certification returns 0. All points receiving 1 for certifications (darker regions) are at least ε-away from the other classes in (a)-(c), i.e. certification is sound. This property is violated in (d), e.g. points from dark red regions are not ε-away from the blue region in the zoomed-in view on the left, but preserved in (e). Namely, voting ensembles are soundness-preserving while cascading ensembles are not.
</p>


## What is in this repository?
### Toy examples
`example/TwoMoon.ipynb` generates a two-dimensional toy example on different ensemble strategies for visualization. 

### Models
`/models/seq_trained/``: models pre-trained by Wong et al. under cascade training strategy. 
`/models/non_seq_trained/`: models trained by use in a non-sequential manner. 

### Re-train the models

#### Pre-requisite

We use the following docker image from [NVIDIA](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) to train the model.
```
nvcr.io/nvidia/pytorch:21.02-py3
```

Some extra packages that do not come with the docker can be installed with `pip`. 

```
pip install scriptify
```

The computation of voting weights is done by using tensorflow. Therefore, you might need Tensorflow 2.x to run it. We will provide a pytorch implementation in the future. 

#### Scripts

`train.sh` includes all the hyper-parameters and instructions needed for non-sequential training of a new model, given the model type and epsilon value.  
`example/mnist.py` and `example/cifar.py` are used for training a new model. 

### Model evaluation
`eval.sh` includes all the hyper-parameters and instructions needed for evaluating a given model. 
`example/evaluate.py` is used for evaluation. 
For a given model evaluatued using either train or test dataset, it generates data files with one row for every sample in the dataset of the form  `(index , predicted label, correct label, is certified?)`
The pre-generated evaluation results are saved in the `evalData` folder. 

### Ensemble
`example/voting.py` uses the evaluated results of the models in the `evalData' folder to calculate the ensemble results based on different ensemble strategies. 
The available strategies include Cascading (unsound), Uniform Voting (sound), and Weighted Voting (sound). 

### More files
All the other files used for training and evaluating the models come from the GitHub page of "Provably robust neural networks" by Wong et al. at https://github.com/locuslab/convex_adversarial/tree/2cd8149249b9e90383af10fc7e9b9fe90166813e.
