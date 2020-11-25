# Decription
Here you can find realisation of Hyperparameter optimization algorithm based on combining of prior and posterior knowledge.

# Running model
## Requirements
Model requires ```python>=3.7``` and corresponding ```pip3```.
First of all install [RoBO](http://automl.github.io/RoBO/installation.html).

## Installation
```datasets-links.txt``` -- file with links to datasets from [OpenML](https://www.openml.org/home) needet for learning.
```datasets-info.csv``` -- file with information about datasets (number of classes, features, instances, e.t.c.).

## File system
Datasets should exist in folder ```./datasets``` int ```.csv``` format.
Training log you can find in _\<date-time\>-training.log_.<br>
In ```optimization_results``` you will find training results and state after each iteration in format _/optimization_results/\<maximizer\>-\<acquisition_function\>-\<model_type\>/dataset_name/_, where:<br>
_\<maximizer\>_ = {"random", "scipy", "differential_evolution"}<br>
_\<acquisition_function\>_ = {"ei", "log_ei", "lcb", "pi"}<br>
_\<model_type\>_ = {"gp", "gp_mcmc", "rf", "bohamiann", "dngo"}<br>
Training can be restored from any saved iteration.
PNG plots will be in ```.png``` folder after each learning session.
