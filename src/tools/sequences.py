# Last editted 2/19/2022 by Benedikt Arnarsson
# Code from https://github.com/JoelForamitti/agentpy

import itertools
import numpy as np
from collections.abc import Sequence

from .Agent import Agent

###########################
# sequences - File for the sequence classes used throughout
#		Includes:
#			- SequenceError (line 23)
# 			- AgentSequence (line 28)
#			- AttrIter (line 65)
#			- AgentList (line 168)
#			- AgentSet (line 225)
#			- AgentIter (line 237)
#			- _random (line 268)
#
###########################

# Simple sequence class
class SequenceError(Exception):
  pass


###########################
# AgentSequence - Base class for sequences (more-so a template/interface)
#
###########################
class AgentSequence:

	def __repr__(self):
		len_ = len(list(self))
		s = 's' if len_ != 1 else ''
		return f"{type(self).__name__} ({len_} objects{s})"
	
	def __getattr__(self, name):
		return AttrIter(self, attr=name)
	
	def _set(self, key, value):
		object.__setattr__(self, key, value)
	
	@staticmethod
	def _obj_gen(model, n, cls, *args, **kwargs):

		if cls is None:
			cls = Agent()
		
		if args != tuple():
			raise SequenceError(
				"Sequences do not accept extra arguments without a keyword"
				f" Please assign a keyword to the following arguments: {args}"
			)
		
		for i in range(n):
			i_kwargs = {k: arg[i] if isinstance(arg, AttrIter) else arg for k, arg in kwargs.items()}
			yield cls(model, **i_kwargs)


###########################
# AttrIter - Building towards an attribute list
#	  - Allows calls, boolean operators, and arithmetic operators
#		to automatically apply per-entry
#	  - Mainly used to set attributes within AgentLists
#
###########################
class AttrIter(AgentSequence, Sequence):

	def __init__(self, source, attr=None):
		self.source = source
		self.attr = attr
	
	def __repr__(self):
		return repr(list(self))
	
	@staticmethod
	def _iter_attr(a, s):
		for o in s:
			yield getattr(o, a)
	
	def __iter__(self):
		if self.attr:
			return self._iter_attr(self.attr, self.source)
		else:
			return iter(self.source)
	
	def __len__(self):
		return len(self.source)
	
	def __getitem__(self, key):
		if self.attr:
			return getattr(self.source[key], self.attr)
		else:
			return self.source[key]
	
	def __setitem__(self, key, value):
		if self.attr:
			setattr(self.source[key], self.attr, value)
		else:
			self.source[key] = value
	
	# Help for function calls (iterate over)
	def __call__(self, *args, **kwargs):
		return AttrIter([func_obj(*args, **kwargs) for func_obj in self])
	
	# Comparison operators
	def __eq__(self, other):
		return [obj == other for obj in self]
	
	def __ne__(self, other):
		return [obj != other for obj in self]
	
	def __lt__(self, other):
		return [obj < other for obj in self]
	
	def __le__(self, other):
		return [obj <= other for obj in self] 
	
	def __gt__(self, other):
		return [obj > other for obj in self]
	
	def __ge__(self, other):
		return [obj >= other for obj in self]

	# Arithmetic operators
	def __add__(self, v):
		if isinstance(v, AttrIter):
			return AttrIter([x + y for x, y in zip(self, v)])
		else:
			return AttrIter([x + v for x in self])
	
	def __sub__(self, v):
		if isinstance(v, AttrIter):
			return AttrIter([x - y for x, y in zip(self, v)])
		else:
			return AttrIter([x - v for x in self])
	
	def __mul__(self, v):
		if isinstance(v, AttrIter):
			return AttrIter([x * y for x, y in zip(self, v)])
		else:
			return AttrIter([x * v for x in self])

	def __truediv__(self, v):
		if isinstance(v, AttrIter):
			return AttrIter([x / y for x, y in zip(self, v)])
		else:
			return AttrIter([x / v for x in self])
	
	def __iadd__(self, v):
		return self + v
	
	def __isub__(self, v):
		return self - v

	def __imul__(self, v):
		return self * v
	
	def __itruediv__(self, v):
		return self / v


###########################
# AgentList -
# 	List of Agent objects
#
#	Arguments:
# 		model (Model): The model instance.
# 		objs (int or Sequence, optional):
#     		An integer number of new objects to be created,
#     		or a sequence of existing objects (default empty).
# 		cls (type, optional): Class for the creation of new objects.
# 		**kwargs:
#     		Keyword arguments are forwarded
#     		to the constructor of the new objects.
#     		Keyword arguments with sequences of type :class:`AttrIter` will be
#     		broadcasted, meaning that the first value will be assigned
#     		to the first object, the second to the second, and so forth.
#     		Otherwise, the same value will be assigned to all objects.
#
###########################
class AgentList(AgentSequence, list):
	def __init__(self, model, objs=(), cls=None, *args, **kwargs):
		if isinstance(objs, int):
			objs = self._obj_gen(model, objs, cls, *args, **kwargs)
		super().__init__(objs)
		super().__setattr__('model', model)
	
	def __setattr__(self, name, value):
		if isinstance(value, AttrIter):
			for obj, v in zip(self, value):
				setattr(obj, name, v)
		else:
			for obj in self:
				setattr(obj, name, value)
	
	def __add__(self, other):
		agents = AgentList(self.model, self)
		agents.extend(other)
		return agents 
	
	# Return a new AgentList based on selection
	# selection - bool[] of same length as self
	def select(self, selection):
		return AgentList(self.model, [a for a, s in zip(self, selection) if s])
	
	def random(self, n=1, replace=False):
		return _random(self.model, self.model.random, self, n, replace)
	
	def sort(self, var_key, reverse=False):
		super().sort(key=lambda x: x[var_key], reverse=reverse)
		return self 
	
	def shuffle(self):
		self.model.random.shuffle(self)
		return self


###########################
# AgentSet -- unordered collection of Agent objects
#
###########################
class AgentSet(AgentSequence, set):
	def __init__(self, model, objs=(), cls=None, *args, **kwargs):
		if isinstance(objs, int):
			objs = self._obj_gen(model, objs, cls, *args, **kwargs)
		super().__init__(objs)
		super().__setattr__('model', model)

###########################
# AgentIter - Iterator over Agent objects
#
###########################
class AgentIter(AgentSequence):
	def __init__(self, model, source=()):
		object.__setattr__(self, '_model', model)
		object.__setattr__(self, '_source', source)
	
	def __getitem__(self, item):
		raise SequenceError('AgentIter has to be converted to list for item lookup')
	
	def __iter__(self):
		return iter(self._source)
	
	def __len__(self):
		return len(self._source)
	
	def __setattr__(self, name, value):
		if isinstance(value, AttrIter):
			for obj, v in zip(self, value):
				setattr(obj, name, v)
		else:
			for obj in self:
				setattr(obj, name, value)
	
	# Convert from AgentIter to AgentList
	def to_list(self):
		return AgentList(self._model, self)


###########################
# _random - Creates a random sample of agents
# 		Arguments:
# 			n (int, optional): Number of agents (default 1).
# 			replace (bool, optional):
# 				Select with replacement (default False).
# 				If True, the same agent can be selected more than once.
# 		Returns:
# 			AgentIter: The selected agents.
# 		
###########################
def _random(model, gen, obj_list, n=1, replace=False):
	if n == 1:
		selection = [gen.choice(obj_list)]
	elif replace is False:
		selection = gen.sample(obj_list, k=n)
	else:
		selection = gen.choices(obj_list, k=n)
	return AgentIter(model, selection)