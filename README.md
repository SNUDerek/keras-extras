# keras-extras
custom scripts for adding functionality to keras

## installation

clone repo and:

use `pip install .` from root dir  
use `pip install -e .` for symlink (updates immediately accessible)

## cross-validation script for keras models

located in `keras-extras.model_selection.KerasGridSearchCV`

this is a cross-validation script for `keras` models that extends functionality beyond the `classifier` and `regressor` `sklearn` wrappers. it will optimize any keras `model` with optional custom evaluation function. it allows for optimizing models beyond simple regessors and classifiers such as models that output sequential outputs, or models with multiple inputs and/or outputs. it also allows custom evaluation functions; e.g. choosing a model with the best recall, or F1 score over a certain subset of classes (instead of just best accuracy).

### notes

- because of the flexibility in output labels, the folds cannot be split in a balanced fashion ('stratified' k-fold)
- right now the best-trained weights are not saved, only the parameters (requires retraining over all data)
- evaluation log cannot be saved to text file, only printed to console

### how to use:

### 1. define a hyperparameter dictionary

this should be a dictionary of hyperparameters with keys as strings and values as lists of possible parameter values. for verbose output, you can define CONSTANTS in UPPERCASE, which will not be printed. *all values* must be a `list`, even constants, which will be a single-element list.

UPDATE: also accepts a list of dictionaries, where each dictionary defines one trial - for running only some experiments.

Use this in the following function:

### 2. define a model generation function

inputs : `config` dictionary
output : a compiled keras `model` file

define your model architecture here, using the keys from the parameter dictionary above.

e.x. `max_len=config['MAX_LEN']` to define the max length of a sequence

### 3. (optional) define a custom evaluation function

inputs : `keras` model, val inputs, val targets, `kwargs` ( *must accept `**kwargs` !* )
output : scalar score

this must accept `**kwargs`; this allows passing an item such as a `tokenizer` to convert model predictions to labels in order to use `sklearn` functions such as calculating `f1_score` on a custom subset of labels.

the basic pattern should be to `model.predict(val_inputs)` and compare with the `val_targets` using a custom metric (recall, precision, F1 score, etc). 

***the score must be a higher-is-better scalar!!!*** for for example, if using loss or another lower-is-better metric, return the negative.

### 4. instantiate a `KerasGridSearchCV` object

initialize the object with the model generator function, the parameter dictionary, and optionally the eval function as `eval_model=myEvalFn`

other parameters are:  
- `epochs` : max epochs to train. overridden by 'epochs' key
- `k` : number of folds for k-fold cross-validation
- `verbose` : boolean; whether to print log to console
- `k_verbose` : the `verbose` setting for `keras` model fitting
- `save_best` : save model weights and log file at each top-scoring model

also, pass any extra things for evaluation using `**kwargs` such as tokenizers.

### 5. use `fit()` to do cross-validation

if using *multi-input* or *multi-output* models, pass the x and/or y ( *training* ) data as a list or `np.array` s: `[x1, x2]`.

you have the opportunity to adjust the`epochs`, `verbose` and `k_verbose` parameters here.

code references:  
[stackexchange: permuting dictionary of lists](https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists)  
[lilly's blog: k-fold cross-validation with sklearn](http://thelillysblog.com/2017/08/18/machine-learning-k-fold-validation/)