# Last editted 3/19/2022 by Benedikt Arnarsson
# Code from https://github.com/JoelForamitti/agentpy

from .Object import Object

###########################
## Agent - a general agent class
##    - given a Model, creates an appropriate agent
##      (with the appropriate variables/parameters)
##
###########################
class Agent(Object):
	def __init__(self, model, *args, **kwargs):
		super().__init__(model)
		self.setup(*args, **kwargs)