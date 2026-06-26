# Last editted 3/29/2022 by Benedikt Arnarsson
# Code from https://github.com/JoelForamitti/agentpy

import pandas as pd
import random as rd
from os import sys
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from tqdm import tqdm
# import winsound
from .helpers import make_list, tqdm_joblib
from .DataDict import DataDict
from .sample import Sample, Range, IntRange, Values

###########################
# Experiment - used to run multiple instances of a model
#              with different parameters (using .sample)
#
###########################
class Experiment:
	def __init__(self, model_class, sample=None, iterations=1, record=False, randomize=True, **kwargs):
		self.model = model_class
		self.output = DataDict()
		self.iterations = iterations
		self.record = record
		self._model_kwargs = kwargs
		self.name = model_class.__name__

		# Prepare sample
		if isinstance(sample, Sample):
			self.sample = list(sample)
			self._sample_log = sample._log
		else:
			self.sample = make_list(sample, keep_none=True)
			self._sample_log = None 
		
		# Prepare runs
		len_sample = len(self.sample)
		iter_range = range(iterations) if iterations > 1 else [None]
		sample_range = range(len_sample) if len_sample > 1 else [None]
		self.run_ids = [(sample_id, iteration)
						for sample_id in sample_range
						for iteration in iter_range]
		self.n_runs = len(self.run_ids)

		# Prepare seeds
		if randomize and sample is not None and any(['seed' in p for p in self.sample]):
			if len_sample > 1:
				rngs = [rd.Random(p['seed'])
						if 'seed' in p else rd.Random() for p in self.sample]
				self._random = {
					(sample_id, iteration): rngs[sample_id].getrandbits(128)
					for sample_id in sample_range
					for iteration in iter_range
				}
			else:
				p = list(self.sample)[0]
				seed = p['seed']
				ranges = (Range, IntRange, Values)
				if isinstance(seed, ranges):
					seed = seed.vdef
				rng = rd.Random(seed)
				self._random = {
					(None, iteration): rng.getrandbits(128)
					for iteration in iter_range
				}
		else:
			self._random = None
		
		# Prepare output
		self.output.info = {
			'model_type': model_class.__name__,
			'time_stamp': str(datetime.now()),
			'python_version': sys.version[:5],
			'experiment': True,
			'scheduled_runs': self.n_runs,
			'completed': False,
			'random': randomize,
			'record': record,
			'sample_size': len(self.sample),
			'iterations': iterations
		}
		self._parameters_to_output()
	
	def _parameters_to_output(self):
		df = pd.DataFrame(self.sample)
		df.index.rename('sample_id', inplace=True)
		fixed_pars = {}
		for col in df.columns:
			s = df[col]
			if len(s.unique()) == 1:
				fixed_pars[s.name] = df[col][0]
				df.drop(col, inplace=True, axis=1)
		self.output['parameters'] = DataDict()
		if fixed_pars:
			self.output['parameters']['constants'] = fixed_pars
		if not df.empty:
			self.output['parameters']['sample'] = df
		if self._sample_log:
			self.output['parameters']['log'] = self._sample_log
		
	@staticmethod
	def _add_single_output_to_combined(single_output, combined_output):
		for k, v in single_output.items():
			if k in ['parameters', 'info']:
				continue
			if isinstance(v, DataDict):
				if k not in combined_output:
					combined_output[k] = {}
				for obj_type, obj_df in single_output[k].items():
					if obj_type not in combined_output[k]:
						combined_output[k][obj_type] = []
					combined_output[k][obj_type].append(obj_df)
			else:
				if k not in combined_output:
					combined_output[k] = []
				combined_output[k].append(v)
	
	def _combine_dataframes(self, combined_output):
		for key, values in combined_output.items():
			if values and all([isinstance(value, pd.DataFrame) for value in values]):
				self.output[key] = pd.concat(values)
			elif isinstance(values, dict):
				self.output[key] = DataDict()
				for sk, sv in values.items():
					if all([isinstance(v, pd.DataFrame) for v in sv]):
						self.output[key][sk] = pd.concat(sv)
					else:
						self.output[key][sk] = sv
			elif key != 'info':
				self.output[key] = values
	
	def _single_sim(self, run_id):
		sample_id = 0 if run_id[0] is None else run_id[0]
		parameters = self.sample[sample_id]
		model = self.model(parameters, _run_id=run_id, **self._model_kwargs)
		if self._random:
			results = model.run(display=False, seed=self._random[run_id])
		else:
			results = model.run(display=False)
		if 'variables' in results and self.record is False:
			del results['variables']
		return results
	
	def run(self, n_jobs=1, display=True, **kwargs):
		if display:
			n_runs = self.n_runs
			print(f"Scheduled runs: {n_runs}")
		t0 = datetime.now()
		combined_output = {}

		if n_jobs != 1:
			with tqdm_joblib(tqdm(desc="Experiment progress", total=self.n_runs)) as progress_bar:
				output_list = Parallel(n_jobs=n_jobs, **kwargs)(
					delayed(self._single_sim)(i) for i in self.run_ids
				)
			for single_output in output_list:
				self._add_single_output_to_combined(
					single_output,
					combined_output
				)
		else:
			i = -1
			for run_id in self.run_ids:
				self._add_single_output_to_combined(
					self._single_sim(run_id), combined_output
				)
				if display:
					i += 1
					td = (datetime.now() - t0).total_seconds()
					te = timedelta(seconds=int(td / (i + 1) * (n_runs - i - 1)))
					print(f"\rCompleted: {i + 1}, estimated time remaining: {te}", end='')
			if display:
				print("")
		
		self._combine_dataframes(combined_output)
		self.end()
		self.output.info['completed'] = True
		self.output.info['run_time'] = ct = str(datetime.now() - t0)

		if display:
			print(f"Experiment finished\nRun time: {ct}")
# 		winsound.Beep(440, 2000)
		
		return self.output

	# Overwrite for final calculations and reporting
	def end(self):
		pass