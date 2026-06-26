# Last editted 3/19/2022 by Benedikt Arnarsson
# Code from https://github.com/JoelForamitti/agentpy

from .helpers import make_list

###########################
## Object - Base class for building modular agents/models/etc. 
##
###########################
class Object:

	# Initialize general framework
	def __init__(self, model):
		self._var_ignore = []

		self.id = model._new_id()
		self.type = type(self).__name__
		self.log = {}

		self.model = model
		self.p = model.p

	# Setting print functionality
	def __repr__(self):
		return f"{self.type} (Obj {self.id})"

	# Raise error if Object doesn't have a feature variable
	# Will be overridden when extending Object
	def __getattr__(self, key):
		raise AttributeError(f"{self} has no attribute '{key}'.")

	# Allows us to do "dictionary-style" lookup of feature variables
	def __getitem__(self, key):
		return getattr(self, key)

	# This allows us to add/set custom variables
	def __setitem__(self, key, value):
		setattr(self, key, value)

	# Store current variables to separate them from custom variables
	def _set_var_ignore(self):
		self._var_ignore = [k for k in self.__dict__.keys() if k[0] != '_']

	@property
	def vars(self):
		return [k for k in self.__dict__.keys()
				if k[0] != '_'
				and k not in self._var_ignore]
	
	# Adding an Object's current (time-step) variables to the log, for later access
	# This is the initial call -- later calls done by self._record
	def record(self, var_keys, value=None):

		# connecting model._logs and self.log
		if self.type not in self.model._logs: 
			self.model._logs[self.type] = {}
		self.model._logs[self.type][self.id] = self.log #setting the reference
		self.log['t']  = [self.model.t] #setting time

		# Initial recording
		for var_key in make_list(var_keys):
			v = getattr(self, var_key) if value is None else value
			self.log[var_key] = [v]
		
		# Setting future record calls
		self.record = self._record

	def _record(self, var_keys, value=None):
		for var_key in make_list(var_keys):
			# Adding new variables
			if var_key not in self.log:
				self.log[var_key] = [None] * len(self.log['t'])
			
			# Checking the time step
			if self.model.t != self.log['t'][-1]:
				for v in self.log.values():
					v.append(None)
				self.log['t'][-1] = self.model.t

			if value is None:
				v = getattr(self, var_key)
			else:
				v = value
			
			self.log[var_key][-1] = v
	
	# Main function for setting parameters and variables of extended classes
	def setup(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)