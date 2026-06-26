# Setting up for experiments (don't change this cell -- default parameters shouldn't change)
from src.caries_20220319 import CariesV2
from src.tools.Experiment import Experiment
from src.tools.sample import Range, Values, Sample

from src.tools.DataDict import DataDict


parameters = {
        # Number of kids
        'agents': 2500,
        # Number of steps in the simulation
        'steps': 36,
        # Model will record every n-th step
        'record_steps': 1,
        # Random seed - can be set to None
        'seed': 17,
        # Main decay/filled/missing/sound parameters
        'person_decay_prob': 0.01356,
        'person_filling_prob': 0.00996,
        'sound_to_cavitated': 0.008994,
        'cavitated_to_missing': 0.00848,
        'person_decay_feedback': 0.25,
        # Fluoride parameters
        'fluoride_effect': 1.00,
        'fluoride_slope': 3.00,
        'fluoride_months_effective': 6.00,
        'person_preventive_fluoride': 1.00,
        'fluoride_scenario': 3.00,
        # High-risk Kid initialization parameters
        'init_caries_prevalence': 0.215311,
        'init_caries_prob': 0.16,
        'init_missing_prob': 0.00848,
        # Behavioral decay parameters
        'behavioral_decay': 2,
        'behavioral_feedback': 1,
        'disable_decay': 0,
    }
    
# Experiment name
experiment_name = 'figuredata_final_run_filled_missing_2500agents_30reps'
num_agents = 2500

# Iterations per sample
num_iterations = 30 
num_jobs = 1 #changes number of parallel runs (cpu usage)

num_steps = 51 # number of samples (should maybe add external step-size function)

exp_parameters = parameters.copy()



# Change number of steps between recording
exp_parameters['record_steps'] = 36 # This means the model will record every 36 steps (only at the beginning/end)

# Change this for percentage of initial high risk
# exp_parameters['init_caries_prevalence'] = 1.00 # all high risk

# Setting behavioral decay to the old default
exp_parameters['behavioral_decay'] = -1 

# Changing fluoride settings
exp_parameters['fluoride_scenario'] = 2.00 
exp_parameters['person_preventive_fluoride'] = 0.1 # Talk about this value
exp_parameters['fluoride_effect'] = 0.1 # and this value

# Changing the number of agents (faster run-time)
exp_parameters['agents'] = num_agents

# Setting the random seed
exp_parameters['seed'] = None

range_filling_prob = Values(0.005, 0.00996, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
# range_missing_prob = Range(vmin=0, vmax=0.2)

exp_parameters['person_filling_prob'] = range_filling_prob
# exp_parameters['cavitated_to_missing'] = range_missing_prob
# exp_parameters['cavitated_to_missing'] =  0.0848

exp_samples = Sample(parameters=exp_parameters, n=num_steps)

# Don't run this cell unless doing an experiment (experiments can take a while)
exp = Experiment(CariesV2, sample=exp_samples, iterations=num_iterations, record=True)
results = exp.run(n_jobs=num_jobs)
results.save(exp_name=experiment_name)

# Baseline
#experiment_name = 'final_run_filled_missing_2500agents_30reps_baseline'
#exp_parameters['person_filling_prob'] = 0.00996
#exp_samples = Sample(parameters=exp_parameters, n=1)
# Don't run this cell unless doing an experiment (experiments can take a while)
#exp = Experiment(CariesV2, sample=exp_samples, iterations=num_iterations, record=True)
#results = exp.run(n_jobs=num_jobs)
#results.save(exp_name=experiment_name)


# Running with no decay
experiment_name = 'figuredata_final_run_filled_missing_2500agents_30reps_nofeedback'
exp_parameters['disable_decay'] = 1
exp_parameters['person_filling_prob'] = range_filling_prob
exp_samples = Sample(parameters=exp_parameters, n=num_steps)
exp = Experiment(CariesV2, sample=exp_samples, iterations=num_iterations, record=True)
results = exp.run(n_jobs=num_jobs)
results.save(exp_name=experiment_name)

# Baseline
#experiment_name = 'final_run_filled_missing_2500agents_30reps_baseline_nodecay'
#exp_parameters['person_filling_prob'] = 0.00996
#exp_samples = Sample(parameters=exp_parameters, n=1)
# Don't run this cell unless doing an experiment (experiments can take a while)
#exp = Experiment(CariesV2, sample=exp_samples, iterations=num_iterations, record=True)
#results = exp.run(n_jobs=num_jobs)
#results.save(exp_name=experiment_name)