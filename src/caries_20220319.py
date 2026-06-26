# Last editted 3/20/2022 by Benedikt Arnarsson
# New version of Caries model
import os
from collections import deque
import pandas as pd

from .tools.ModelFrame import ModelFrame
from .tools.sequences import AgentList
from .CariesKid import Kid

class CariesV2(ModelFrame):
	kid_variables = {
		"high_risk": 0,
		"caries_xp": 0,
		"caries": 0,
		"missing": 0,
		"filled": 0,
		"sound": 0,
		"fluoride_val": 0,
		"fluoride_time": 0,
    "untreated_time": 0,
	}	
	
	def setup(self):
		# Setting up the variables
		# if 'variables' in self._setup_kwargs:
		# 	for k, v in self._setup_kwargs['variables'].items():
		# 		setattr(self, k, v)

		# Setting up the reporters
		# if 'reporters' in self._setup_kwargs:
		# 	for k, v in self._setup_kwargs['reporters'].items():
		# 		self.reporters.update(k, v)
		# 		self._var_ignore.append(k)
		# 		setattr(self, k, v)

		# Creating list of agents
		agent_init_par = {
			"variables": self.kid_variables,
			"caries_prev": self.p.init_caries_prevalence,
			"caries_prob": self.p.init_caries_prob,
			"missing_prob": self.p.init_missing_prob,
		}
		self.agents = AgentList(self, self.p.agents, Kid, **agent_init_par)

		# Counting high and low risk
		num_hi_risk = len(self.agents.select(self.agents.high_risk))
		self.count_hi_risk = float(num_hi_risk)
		self.count_lo_risk = self.p.agents - num_hi_risk

		# Cost analysis output
		self.cost_csv_name = 'cost_analysis.csv'
		self.run_id = 0
		if os.path.exists(self.cost_csv_name):
			with open(self.cost_csv_name) as cost_file:
				last_line = deque(cost_file)[0]
				last_run_id = last_line.split(',')[1]
				try:
					self.run_id = int(last_run_id) + 1
				except:
					self.run_id = 0
		self.cost_columns = [
			'Model Run',
			'AgentId',
			'High/Low',
			'Current Timestep',
			'Event type',
			'Time Decayed',
			'Fill Probability',
			'Missing Probability',
		]
		self.cost_data = []


	
	def step(self):
		# Filled/missing step
		self.agents._get_treatment()

		# can't get cavities on filled teeth
		self.agents._get_cavities()

        # Fluoride step
		self.agents._fluoride_step()
	
	def update(self):
		for agent in self.agents:
			agent.untreated_time += agent.caries
        
		if self.t%self.p.record_steps == 0 or self.t == self.p.steps:
			for var_key in self.kid_variables:
				self.agents.record(var_key)

		# self.agents.record("sound")
		# self.agents.record("caries")
		# self.agents.record("missing")
		# self.agents.record("caries_xp")

		# self.record(var_keys=self.vars)
	
	def create_output(self):
		super().create_output()
		cost_df = pd.DataFrame(self.cost_data, columns=self.cost_columns)
		# Record cost related data
		if not os.path.exists('./cost_analysis.csv'):
			cost_df.to_csv('./cost_analysis.csv')
		else:
			cost_df.to_csv('cost_analysis.csv', mode='a', header=False)

	def end(self):
		if bool(self.reporters):
			self.report(rep_keys=list(self.reporters.keys()))