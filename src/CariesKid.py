# Last editted 3/20/2022 by Benedikt Arnarsson
# New person model

import math
import numpy as np
from .tools.Agent import Agent

class Kid(Agent):

	# Initializing our Kid agents
	def setup(self, variables, caries_prev, caries_prob, missing_prob):		
		super().setup(**variables)
		self._decayed_teeth_time = []
		self._var_ignore.append('_decayed_teeth_time')
		self._set_hi_risk(caries_prev, caries_prob, missing_prob)
	
	# Randomly setting high_risk
	def _set_hi_risk(self, caries_prev, caries_prob, missing_prob):
		tmp_rnd = self.model.nprandom.random()
		if tmp_rnd < caries_prev:
			self.high_risk = 1
			# Giving the first cavity
			self.caries_xp = 1
			self.caries += 1
			self._decayed_teeth_time.append(0)
			for n in range(1, 20):
				caries_rnd = self.model.nprandom.random()
				if caries_rnd <= caries_prob:
					self.caries += 1
					self._decayed_teeth_time.append(0)
				elif caries_rnd <= (caries_prob + missing_prob):
					self.missing += 1
				else:
					self.sound += 1
			if self.p.fluoride_scenario == 1.00:
				self.fluoride_time = 1.00
				self.fluoride_val = self.p.fluoride_effect
		else:
			self.sound += 20

	######################
	# Defining different parts of the step()
	#
	######################
	# Randomly giving treatment for cavities (or losing teeth)
	def _get_treatment(self):
		new_caries = 0
		new_filled = 0
		new_missing = 0

		kid_fillings = self.model.nprandom.random()
		
		if kid_fillings <= self.p.person_filling_prob:
			new_caries -= self.caries
			new_filled +=  self.caries
			for _ in range(new_filled):
				self.model.cost_data.append([
					self.model.run_id,
					self.id,
					'High' if self.high_risk == 1 else 'Low',
					self.model.t,
					'Filled',
					self.model.t - self._decayed_teeth_time.pop(0),
					self.p.person_filling_prob,
					self.p.cavitated_to_missing,
				])
			if self.p.fluoride_scenario == 2.00:
				self.fluoride_time = 1.00
				self.fluoride_val = self.p.fluoride_effect
		else:
			missing_prob = self.p.cavitated_to_missing
			if self.caries > 0:
				for _ in range(self.caries):
					tooth_missing = self.model.nprandom.random()
					# missing_prob = missing_prob + np.divide(1 + np.log(v), 10)
					if tooth_missing <= missing_prob:
						new_caries -= 1
						new_missing += 1
						self.model.cost_data.append([
							self.model.run_id,
							self.id,
							'High' if self.high_risk == 1 else 'Low',
							self.model.t,
							'Missing',
							self.model.t - self._decayed_teeth_time.pop(0),
							self.p.person_filling_prob,
							self.p.cavitated_to_missing,
						])
		
		self.caries +=	new_caries
		self.filled += new_filled
		self.missing += new_missing

	# Randomly giving our agent cavities
	def _get_cavities(self):
		new_caries = 0
		new_sound = 0

		kid_decay = self.model.nprandom.random()
		preventive_fluoride = self.model.nprandom.random()

		decay_prob = self.p.person_decay_prob
		if preventive_fluoride <= self.p.person_preventive_fluoride:
			self.fluoride_time = 1
			self.fluoride_val = self.p.fluoride_effect

		# used to be 
		# num_decay = self.caries + self.missing
		#	-- put for behavioral reasons
		#	-- could include caries_xp or high_risk
		num_decay = 0
		if self.p.behavioral_decay==-1: # The first model scenario
			num_decay = self.caries + self.missing
		else:
			if self.caries == 0:
				if self.p.behavioral_decay==1 and self.caries_xp==1:
					num_decay = self.p.behavioral_feedback
				elif self.p.behavioral_decay==2 and self.high_risk==1:
					num_decay = self.p.behavioral_feedback
				elif self.p.behavioral_decay==3:
					num_decay = self.missing
			else:
				num_decay = self.caries
		if num_decay > 0:
			decay_prob = self.p.person_decay_prob + np.divide(1 + np.log(num_decay) * self.p.person_decay_feedback, 10)

		decay_prob = decay_prob - (decay_prob * self.fluoride_val)
		if self.p.disable_decay == 1:
			decay_prob = self.p.person_decay_prob
			
		if kid_decay <= decay_prob:
			if self.caries_xp == 0:
				self.caries_xp = 1
			at_least_one = False
			for k in range(self.sound):
				if not at_least_one:
					new_sound -= 1
					new_caries += 1
					at_least_one = True
				else:
					rnd = self.model.nprandom.random()
					if rnd <= self.p.sound_to_cavitated:
						new_sound -= 1
						new_caries += 1
		
		self.sound += new_sound
		self.caries += new_caries
		if new_caries > 0:
			for _ in range(new_caries):
				self._decayed_teeth_time.append(self.model.t)
	
	# Updating fluoride variables
	def _fluoride_step(self):
		if self.fluoride_time > self.p.fluoride_months_effective:
			if self.p.fluoride_scenario == 1.00:
				self.flouride_val = self.p.fluoride_effect
				self.fluoride_time = 1.00
			else:
				self.fluoride_val = 0.00
				self.fluoride_time = 0.00
		elif self.fluoride_time > 0.00:
			self.fluoride_time += 1.00
			self.fluoride_val = float(self.p.fluoride_effect) / ( 1 + math.exp(float(self.fluoride_time) - self.p.fluoride_slope))
				