# Last editted 3/19/2022 by Benedikt Arnarsson
# Code from https://github.com/JoelForamitti/agentpy

import numpy as np
import pandas as pd
import random

from os import sys
from datetime import datetime

from .Object import Object
from .AttrDict import AttrDict
from .sample import Range, Values
from .sequences import AgentList
from .DataDict import DataDict
from .helpers import make_list, InfoStr

###########################
## ModelFrame - a general model framework
## 
###########################
class ModelFrame(Object):
	def __init__(self, parameters=None, _run_id=None, **kwargs):

		# Prepare parameters
		self.p = AttrDict()
		if parameters:
			for k, v in parameters.items():
				if isinstance(v, (Range, Values)):
					v = v.vdef
				self.p[k] = v
		
		# Initiate model with id 0
		self._id_counter = -1
		super().__init__(self)

		# Simulating attributes
		self.t = 0
		self.running = False
		self._run_id = _run_id

		# Random number generators
		# Seed can be set at Model.run()
		self.random = random.Random()
		self.nprandom = np.random.default_rng()

		# Recording results
		self._logs = {}
		self.reporters = {}
		self.output = DataDict()
		self.output.info = {
			'model_type': self.type,
			'time_stamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
			'python_version': sys.version[:5],
			'experiment': False,
			'completed': False
		}

		# Private variables
		self._steps = None
		self._partly_run = False
		self._setup_kwargs = kwargs
		self._set_var_ignore()
	
	def __repr__(self):
		return self.type
	
	# Getting information
	@property
	def info(self):
		rep = f"Agent-based mode {{"
		items = list(self.__dict__.items())
		for k, v in items:
			if k[0] != '_':
				v = v._short_repr() if '_short_repr' in dir(v) else v
				rep += f"\n'{k}': {v}"
		rep += '\n}'
		return InfoStr(rep)
	
	# Handling model and agent ids
	def _new_id(self):
		self._id_counter += 1
		return self._id_counter
	
	# Recording
	def report(self, rep_keys, value=None):
		for rep_key in make_list(rep_keys):
			if value is not None:
				self.reporters[rep_key] = value 
			else:
				try:
					self.reporters[rep_key] = getattr(self, rep_key)
				except:
					self.reporters[rep_key] = self.p[rep_key]
	
	###########################
	# Placeholder methods for custom simulation
	# 		Change these methods when extending
	#
	###########################
	# Define models actions. Initiate agents and environment
	def setup(self):
		pass
	
	# Define model's action during simulation step
	#  - mainly for defining model dynamics
	def step(self):
		pass

	# Define model's action after simulation step
	#  - mainly for recording dynamic variables
	def update(self):
		pass
	
	# Define model's action after last simulation step
	#  - mainly for final calculations and reporting
	def end(self):
		pass

	###########################
	# Simulation methods
	#
	###########################

	def set_parameters(self, parameters):
		self.p.update(parameters)
	
	def sim_setup(self, steps=None, seed=None):
		# Prepare random number generator (if initial run)
		if self._partly_run is False:
			if seed is None:
				if 'seed' in self.p:
					seed = self.p['seed']
				else:
					seed = random.getrandbits(128)
			if not ('report_seed' in self.p and not self.p['report_seed']):
				self.report('seed', seed)
			self.random = random.Random(seed)
			npseed = self.random.getrandbits(128)
			self.nprandom = np.random.default_rng(seed=npseed)
		
		# Prepare simulation steps
		if steps is None:
			self._steps = self.p['steps'] if 'steps' in self.p else np.nan
		else:
			self._steps = self.t + steps
		
		# Initiate simulation
		self.running = True
		self._partly_run = True
		
		# Execute setup and first update
		self.setup(**self._setup_kwargs)
		self.update()

		# Stop simulation if t is too high
		if self.t >= self._steps:
			self.running = False
	
	def sim_step(self):
		self.t += 1
		self.step()
		self.update()
		if self.t >= self._steps:
			self.running = False
	
	###########################
	# Main methods for running the model
	#
	###########################

	def stop(self):
		self.running = False
	
	# Starts a simulation of the model, or continues _partly_run models
	# Returns a DataDict of variables and reporters
	def run(self, steps=None, seed=None, display=True):
		dt0 = datetime.now()
		self.sim_setup(steps, seed)
		while self.running:
			self.sim_step()
			if display:
				print(f"\rCompleted: {self.t} steps", end='')
		self.end()
		self.create_output()

		self.output.info['completed'] = True
		self.output.info['created_objects'] = self._id_counter
		self.output.info['completed_steps'] = self.t
		self.output.info['run_time'] = ct = str(datetime.now() - dt0)

		if display:
			print(f"\nRun time: {ct}\nSimulation Finished")
		
		return self.output
	
	###########################
	# Data management
	#
	###########################
	#   create_output 
	# 		- sets stores a DataDict with all variables and reporters
	#		  in Model.output
	#
	###########################
	def create_output(self):

		def output_from_obj_list(self, log_dict, columns):
			obj_types = {}
			for obj_type, log_subdict in log_dict.items():
				if obj_type not in obj_types.keys():
					obj_types[obj_type] = {}
				
				for obj_id, log in log_subdict.items():
					# Add object id/key to object log
					log['obj_id'] = [obj_id] * len(log['t'])

					# Add object log to aggregate log
					for k, v in log.items():
						if k not in obj_types[obj_type]:
							obj_types[obj_type][k] = []
						obj_types[obj_type][k].extend(v)
			# Dictionary to DataFrame
			for obj_type, log in obj_types.items():
				if obj_type == self.type:
					del log['obj_id']
					index_keys = ['t']
				else:
					index_keys = ['obj_id', 't']
				df = pd.DataFrame(log)
				for k, v in columns.items():
					df[k] = v
				df = df.set_index(list(columns.keys()) + index_keys)
				self.output['variables'][obj_type] = df
		
		# Step 1: Document parameters
		if self.p:
			self.output['parameters'] = DataDict()
			self.output['parameters']['constants'] = self.p.copy()
		
		# Step 2: Define additional index columns
		columns = {}
		if self._run_id is not None:
			if self._run_id[0] is not None:
				columns['sample_id'] = self._run_id[0]
			if len(self._run_id) > 1 and self._run_id[1] is not None:
				columns['iteration'] = self._run_id[1]
		
		# Step 3: Create variable output
		if self._logs:
			self.output['variables'] = DataDict()
			output_from_obj_list(self, self._logs, columns)
		
		# Step 4: Create reporters output
		if self.reporters:
			d = {k: [v] for k, v in self.reporters.items()}
			for key, value in columns.items():
				d[key] = value 
			df = pd.DataFrame(d)
			if columns:
				df = df.set_index(list(columns.keys()))
			self.output['reporters'] = df
		
