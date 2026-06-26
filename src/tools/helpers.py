# Last editted 4/5/2022 by Benedikt Arnarsson

import numpy as np
from numpy import ndarray
from os import listdir

###########################
# helpers - File of helper functions used throughout the project
#		Includes:
#			- gauss01Redraw (line 18)
#			- divide (line 28)
#			- make_list (line 39)
#			- _last_exp_id (line 56)
#			- InfoStr (line 69)
#
###########################


###########################
## gauss01Redraw -- drawing a value in (0,1) from a Gaussian with a mean of mean and standard deviation of sd
##      - Currently not being used 
###########################
# def gauss01Redraw (mean, sd):
#         m = -1
#         while m > 1 or m < 0:
#             m = random.gauss ( mean, sd )
#         return m

###########################
## divide -- cast two numeric variables as floats and divide
##      - ensures accuracy of division (no integer division)
###########################
def divide( x, y ):
        try:
            return ( float(x) / float(y) )
        except:
            return 0


###########################
## make_list -- params: element (any type), keep_none (boolean)
##    - If element is a not a set or tuple, turns element into a list of itself
##    - If element is a set or tuple, casts them to a list
## 		- Code from https://github.com/JoelForamitti/agentpy
###########################
def make_list(element, keep_none=False):
	if element is None and not keep_none:
		element = [element]
	elif not isinstance(element, (list, tuple, set, ndarray)):
		element = [element]
	elif isinstance(element, (tuple, set)):
		element = list(element)
	
	return element


###########################	
# _last_exp_id - Returns the latest experiment (with the highest id)
#					- Code from https://github.com/JoelForamitti/agentpy
###########################
def _last_exp_id(name, path):
	output_dirs= listdir(path)
	exp_dirs = [s for s in output_dirs if name in s]
	if exp_dirs:
		ids = [int(s.split('_')[-1]) for s in exp_dirs]
		return max(ids)
	else:
		return None

###########################
# InfoStr - display Strings without quotations
#
###########################
class InfoStr(str):
	def __repr__(self):
		return self


###########################
# tqdm_joblib - allows us to show a progress bar when running parallel models
#   - https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
###########################
import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

###########################
# step_size_to_num_steps - helper for experiment_template.ipynb
#       - converts step size and range to number of steps
#       - not in use currently
###########################
def step_size_to_num_steps(step_size, start=0.00, end=1.00):
    # step_size = (end-start)/num_steps
    num_steps = np.divide((end-start), step_size) + 1
    # need to add one to the num_steps because of 0 start index
    return num_steps