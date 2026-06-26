# Last editted 3/19/2022 by Benedikt Arnarsson
# Code from https://github.com/JoelForamitti/agentpy

import itertools
import random
import numpy as np

###########################
# samples -- a set of classes and methods for sample work (used in Experiment)
#	Includes:
#		- Sample (line 19)
# 	- Range (line 83)
#		- IntRange (line 105)
#		- Values (line 121)
#		- SampleError (line 142)
#
###########################

###########################
# Sample - sequence of parameters for use in the Experiment class
#
###########################
class Sample:

	# Class initialization and attributes
	def __init__(self, parameters, n=None, method='linspace', randomize=True, **kwargs):
		self._log = {'type': method, 'n': n, 'randomized': False}
		self._sample = getattr(self, f"_{method}")(parameters, n, **kwargs)
		if 'seed' in parameters and randomize:
			ranges = (Range, IntRange, Values)
			if not isinstance(parameters['seed'], ranges):
				seed = parameters['seed']
				self._log['randomized'] = True
				self._log['seed'] = seed
				self._assign_random_seeds(seed)
	
	def __repr__(self):
		return f"Sample of {len(self)} parameter combinations"
	
	def __iter__(self):
		return iter(self._sample)
	
	def __len__(self):
		return len(self._sample)

	###########################
	# Sample methods
	#   - currently only using linspace from numpy
	###########################
	def _assign_random_seeds(self, seed):
		rng = random.Random(seed)
		for parameters in self._sample:
			parameters['seed'] = rng.getrandbits(128)
	
	@staticmethod
	def _linspace(parameters, n, product=True):
		params = {}
		for k, v in parameters.items():
			if isinstance(v, Range):
				if n is None:
					raise SampleError("Argument 'n' must be defined for Sample if there are parameters of type Range.")
				if v.ints:
					p_range = np.linspace(v.vmin, v.vmax+1, n)
					p_range = [int(pv)-1 if pv == v.max+1 else int(pv) for pv in p_range]
				else:
					p_range = np.linspace(v.vmin, v.vmax, n)
				params[k] = p_range
			elif isinstance(v, Values):
				params[k] = v.values
			else:
				params[k] = [v]

		if product:
			combos = list(itertools.product(*params.values()))
			sample = [{k: v for k,v in zip(params.keys(), c)} for c in combos]
		else:
			r = range(min([len(v) for v in params.values()]))
			sample = [{k: v[i] for k, v in params.items()} for i in r]
		
		return sample


###########################
# Range - A range of parameter values
# 		that can be used to create a :class:`Sample`.
#		Arguments:
#      	vmin (float, optional):
#       	Minimum value for this parameter (default 0).
#       vmax (float, optional):
#           Maximum value for this parameter (default 1).
#       vdef (float, optional):
#           Default value. Default value. If none is passed, `vmin` is used.
#     
###########################
class Range:
	def __init__(self, vmin=0, vmax=1, vdef=None):
		self.vmin = vmin
		self.vmax = vmax
		self.vdef = vdef if vdef else vmin
		self.ints = False
	
	def __repr__(self):
		return f"Parameter range from {self.vmin} to {self.vmax}"

###########################
# IntRange - A range of integer parameter values
# 		that can be used to create a :class:`Sample`.
# 		(sample values rounded and cast to int)
###########################
class IntRange(Range):
	def __init__(self, vmin=0, vmax=1, vdef=None):
		self.vmin = int(round(vmin))
		self.vmax = int(round(vmax))
		self.vdef = int(round(vdef)) if vdef else int(round(vmin))
		self.ints = True
	
	def __repr__(self):
		return f"Integer parameter range from {self.vmin} to {self.vmax}"


###########################
# Values - A pre-defined set of discrete parameter values
# 		that can be used to create a :class:`Sample`.
# 		Arguments:
#     		*args:
#         		Possible values for this parameter.
#     		vdef:
#         		Default value. If none is passed, the first passed value is used.
#
###########################
class Values:
	def __init__(self, *args, vdef=None):
		self.values = args
		self.vdef = vdef if vdef else args[0]
	
	def __len__(self):
		return len(self.values)

	def __repr(self):
		return f"Set of {len(self.values)} parameter values"

###########################
# SampleError - a simple Exception class
#
###########################
class SampleError(Exception):
	pass