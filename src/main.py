# Last editted 3/31/2022 by Benedikt Arnarsson
# Not really using this anymore
#  -- see the .ipynb files in the main folder
# saved so that people can know how to run the first model


###########################
## main -- running the model and saving to "report"
##
###########################
# from first_model.caries_20180211 import caries
from .caries_20220319 import CariesV2
from .tools.Experiment import Experiment
from .tools.sample import Range, Values, Sample

def run_model_1():
    m1 = caries()
    m1.initialize()
    m1.processCmdLineArgs()
    print("processing runm")
    m1.checkAndProcessRunmPars()
    m1.startRNG()
    m1.createStuff()
    m1.openReportFile("report")
    s1 = m1.createReportString()
    m1.writeStringToReport0( 0, s1 ) # init state
    m1.clearAggregates()
    m1.run()               # send it the run msg -- sends one model.step msg
    m1.closeReportFile( "report" )

def run_model_2():
    parameters = {
        # Number of kids
        'agents': 2500,
        # Number of steps in the simulation
        'steps': 36,
        'record_steps': 6,
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
    } 

    # Dry run with base parameters
    model = CariesV2(parameters=parameters)
    results = model.run()
    # results.save()

    # Setting sample ranges for experiment
    range_filling_prob = Range(vmin=0, vmax=1)
    range_missing_prob = Range(vmin=0, vmax=1)
    # range_hi_risk_prob = Values(0, 0.215311, 1)
    # add percentage of high risk: 0%, 100%, and default
    # heatmaps -- check original paper
    # R libraries -- for handling time series

    # filling_parameters = parameters.copy()
    # filling_parameters['person_filling_prob'] = range_filling_prob
    
    # add option for step-size
    # samples_filling = Sample(parameters=filling_parameters, n=10)

    # separate iteration .csv
    # exp_filling = Experiment(CariesV2, sample=samples_filling, iterations=5, record=True)

    # results_filling = exp_filling.run(n_jobs=4)
    # results_filling.save(exp_name='CariesV2_filling')

    # missing_parameters = parameters.copy()
    # missing_parameters['cavitated_to_missing'] = range_missing_prob
    # samples_missing = Sample(parameters=missing_parameters, n=10)
    # exp_missing = Experiment(CariesV2, sample=samples_missing, iterations=5, record=True)
    # results_missing = exp_missing.run(n_jobs=4)
    # results_missing.save(exp_name='CariesV2_missing')

    # This experiment will run 500 (5*10*10) iterations, resulting in over 46 million variables being saved
    # maybe not the best...

    both_parameters = parameters.copy()
    both_parameters['person_filling_prob'] = range_filling_prob
    both_parameters['cavitated_to_missing'] = range_missing_prob
    both_parameters['init_caries_prevalence'] = 1.00
    samples_both = Sample(parameters=both_parameters, n=11)
    exp_both = Experiment(CariesV2, sample=samples_both, iterations=5, record=True)
    results_both = exp_both.run(n_jobs=4)
    results_both.save(exp_name='CariesV2_both_hi')

if __name__=='__main__':
    # run_model_1()
    # run_model_2()
    pass