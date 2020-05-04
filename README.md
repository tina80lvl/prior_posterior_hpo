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
Datasets should exist in folder ```./datasets``` int ```.csv``` format and have namemes *name_train.csv* and *name_test.csv* for each dataset.
PNG plots will be in '''.png''' folder after each learning session.
