# Last editted 3/19/2022 by Benedikt Arnarsson
# Code from https://github.com/JoelForamitti/agentpy

###########################
## AttrDict - a class for managing and storing attributes of our model
##		- attributes referring to parameters, variables, and other features
###########################
class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		if args == (None, ):
			args = ()
		super().__init__(*args, **kwargs)
	
	def __getattr__(self, key):
		try:
			return self.__getitem__(key)
		except:
			raise AttributeError(key)
	
	def __setattr__(self, key, value):
		self.__setitem__(key, value)
	
	def __delattr__(self, key):
		del self[key]
	
	def _short_repr(self):
		len_ = len(self.keys())
		return f"AttrDict ({len_} entr{'y' if len_ == 1 else 'ies'})"