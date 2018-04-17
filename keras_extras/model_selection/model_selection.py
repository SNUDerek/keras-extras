import re
import numpy as np
import itertools
from sklearn.model_selection import KFold

class KerasGridSearchCV:
    """
    cross-validation class for keras model

    Attributes
    ----------
    make_model : function
        function that accepts param dict and outputs model
    param_grid : dict
        dictionary of hyperparam : lists of values (CONSTANTS : len-1 list)
    eval_model : function
        custom eval; should give higher-is-better metric score
    epochs : int
        epochs to train (if 'epochs' not param in grid)
    k : int
        number of folds for cross-validation
    verbose : bool
        print out intermediate information during x-validation
    k_verbose : int
        verbosity setting for keras training (0, 1 or 2)
    """
    
    def __init__(self, make_model, param_grid, eval_model=None, epochs=20, k=5, verbose=True, k_verbose=0, **kwargs):
        self.make_model = make_model
        self.eval_model = eval_model
        self.param_grid = param_grid
        self.k = k
        self.epochs = epochs
        self.verbose = verbose
        self.k_verbose = k_verbose
        self.kwargs = kwargs
        self.kf = KFold(n_splits=self.k, shuffle=True, random_state=1337)
        self.x_train = None
        self.y_train = None
        self.best_params = None
        self.trials = []
        self.scores = []
        
    def accuracy_eval(self, model, x_val, y_val):
        """
        evaluate single-loss fn with metrics=['accuracy]

        Attributes
        ----------
        model : keras.model
            keras model to evaluate
        x_val : np.array
            withheld input data to eval on
        y_val : np.array
            withheld label data to eval on
        """
        scores = model.evaluate(x_val, y_val, verbose=self.k_verbose)
        score = scores[-1]
        if self.verbose:
            print('score:', score)
        return score
        
    def fit(self, x_train, y_train, epochs=None, verbose=None, k_verbose=None):
        """
        do a parameter grid search x-validation with the given data

        Attributes
        ----------
        x_train : np.array
            training input data, used in k-fold cross-validation
        y_train : np.array
            training label data, used in k-fold cross-validation
        epochs : int
            same as init
        verbose : bool
            same as init
        k_verbose : int
            same as init
            
        Returns
        -------
        best_params : dict
            dictionary of the highest-scoring hyperparameters
        trials : list
            list of all parameters tested
        scores : list
            list of all scores (avg'd over folds) corresponding to trials
        """
        self.x_train = x_train
        self.y_train = y_train
        if epochs is not None:
            self.epochs = epochs
        if verbose is not None:
            self.verbose = verbose
        if k_verbose is not None:
            self.k_verbose = k_verbose
        
        # generate every combination of parameters
        keys, values = zip(*self.param_grid.items())
        for i, v in enumerate(itertools.product(*values)):
            
            # get sample params as dict
            trial = dict(zip(keys, v))
            if self.verbose:
                print()
                print()
                print("testing", i+1, ':')
                for k, v in trial.items():
                    if not k.isupper():
                        print(k, ':\t', v)
                print()
            
            # get model
            trial_model = self.make_model(trial)
            if self.verbose:
                # save weights to re-initialize each fold
                winit = trial_model.get_weights()
                # trial_model.summary()
                # print()
                # print()
                
            # k-fold sample
            # handle multi-input format
            if type(x_train) is list:
                x_sample = np.arange(x_train[0].shape[0])
            else:
                x_sample = np.arange(x_train.shape[0])
                
            trial_scores = []
            c = 1
            for train_index, test_index in self.kf.split(x_sample):
                
                # split x data - handle multiple input models
                if type(self.x_train) is list:
                    x_learn = [x[train_index] for x in self.x_train]
                    x_eval = [x[test_index] for x in self.x_train]
                else:
                    x_learn = self.x_train[train_index]
                    x_eval  = self.x_train[test_index]
                
                # split y data - handle multiple input models
                if type(self.y_train) is list:
                    y_learn = [y[train_index] for y in self.y_train]
                    x_eval = [y[test_index] for y in self.y_train]
                else:
                    y_learn = self.y_train[train_index]
                    y_eval  = self.y_train[test_index]

                # reset model
                trial_model.set_weights(winit)
                
                # check for epochs
                if 'epochs' in trial:
                    epochs = trial['epochs']
                else:
                    epochs = self.epochs
                
                # fit
                if self.verbose:
                    print()
                    print('training', c, 'th fold...')
                history = trial_model.fit(x_learn, y_learn, epochs=epochs, verbose=self.k_verbose)
                
                # eval
                if self.verbose:
                    print('evaluating', c, 'th fold...')
                if self.eval_model is not None:
                    score = self.eval_model(trial_model, x_eval, y_eval, **self.kwargs)
                else:
                    score = self.accuracy_eval(trial_model, x_eval, y_eval)
                if self.verbose:
                    print('evaluation score :', score)
                # add
                trial_scores.append(score)
                c += 1
            
            avg_score = sum(trial_scores)/len(trial_scores)
            if self.verbose:
                print()
                print('final avg score :', avg_score)
                                
            # compare and save if best
            if len(self.scores) > 0:
                if avg_score > max(self.scores):
                    self.best_params = trial
                    if self.verbose:
                        print()
                        print("new best model with avg score", avg_score)
            else:
                self.best_params = trial
                if self.verbose:
                    print()
                    print("new best model with avg score", avg_score)
            
            self.trials.append(trial)
            self.scores.append(avg_score)
            
        print()
        print("best params:")
        print(self.best_params)
        
        return self.best_params, self.trials, self.scores