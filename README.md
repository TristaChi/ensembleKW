# On the Perils of Cascading Robust Classifiers
A repository for evaluating the ensemble results under different stratagis for the Kolter Wong models.
See more about the Kolter Wong models here: https://github.com/locuslab/convex_adversarial/tree/2cd8149249b9e90383af10fc7e9b9fe90166813e

## What is this repository used for?
Cascading ensembles are a popular instance of black-box ensembles that appear to improve certified robust accuracies in practice. 
However, we find that the robustness certifier used by a cascading ensemble is unsound. 
Here is the experimental support for cascading ensemble (unsound) and alternative weighted voting ensemble (sound). 
For all ensemble stratagies, we used pre-trained and self-traind Kolter Wong models.

## What is in this repository?
