# On the Perils of Cascading Robust Classifiers
The repository supports the paper "On the Perils of Cascading Robust Classifiers."
This repository evaluates the ensemble results under different strategies for the Kolter Wong models.
See more about the Kolter Wong models here: https://github.com/locuslab/convex_adversarial/tree/2cd8149249b9e90383af10fc7e9b9fe90166813e

## What is this repository used for?
Cascading ensembles are a popular instance of black-box ensembles that appear to improve certified robust accuracies in practice. 
However, we find that the robustness certifier used by a cascading ensemble is unsound. 
The experimental support for cascading ensemble (unsound) and alternatively weighted voting ensemble (sound) is here. 
For all ensemble strategies, we used pre-trained and self-trained Kolter Wong models.

## Visualization of Cascading Ensemble unsoundness

<p align = "center">
<img src = "https://github.com/TristaChi/ensembleKW/blob/main/examples/ensemble.png">
</p>
<p align = "left">
Visualizing classification results of 2D points for constituent models (a-c) and the corresponding Cascading Ensemble (d) and Uniform Voting Ensemble (e). Regions with colors correspond to predictions (0: red, 1: blue, 2: green) made by the underlying model (or ensemble). Darker colors indicate that the accompanying robustness certification of the underlying model (or ensemble) returns 1 and lighter colors are for cases when the certification returns 0. All points receiving 1 for certifications (darker regions) are at least ε-away from the other classes in (a)-(c), i.e. certification is sound. This property is violated in (d), e.g. points from dark red regions are not ε-away from the blue region in the zoomed-in view on the left, but preserved in (e). Namely, voting ensembles are soundness-preserving while cascading ensembles are not.
</p>


## What is in this repository?
### Toy examples
`example/2D.ipynb` and `example/TwoMoon.ipynb` generate two-dimensional toy examples on different ensemble strategies for visualization. 

### Models
`/models/models_scaled/` and `/models/models_scaled_l2/`: models pre-trained by Wong et al. under cascade trainning strategy. 
`/models/more_models/`: non-sequentially self trained models. 

### Re-train the models
`train.sh` includes all the hyper-parameters and instructions needed for training a new model, given the model type and epsilon value.  
`example/mnist.py` and `example/cifar.py` are used for training a new model. 

### Model evaluation
`eval.sh` includes all the hyper-parameters and instructions needed for evaluating a given model. 
`example/evaluate.py` is used for evaluation. 
It generates data files of (index, predict label, correct label, certified) on train and test dataset on a given model.
The pre-generated evaluation results are saved in the `evalData` file. 

### Ensemble
`example/voting.py` is used for finding the ensemble results based on different ensemble strategies on the evaluated results of given models. 
The available strategies include Cascading (unsound), Uniform Voting (sound), and Weighted Voting (sound). 

### More files
All the other files used for training and evaluating the models come from the GitHub page of "Provably robust neural networks" by Wong et al. at https://github.com/locuslab/convex_adversarial/tree/2cd8149249b9e90383af10fc7e9b9fe90166813e.
