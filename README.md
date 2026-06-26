# Caries Dental Model

Welcome to the Caries Dental Model! The goal of this code is to model the behavior of cavity development in young children (ages 1-6). 

That goal of this document is to describe the architecture of this agent-based model so that it can be easily modified at a future date.

----
## Table of Contents

- [Installation](#installation)
- [Code Overview](#code-overview)
- [Quick Usage](#quick-usage)
- [Current Model](#current-model)
- [Output Details](#output-details)

----

## Installation

A conda environment is contained in `environment.yml`. First install a version of Anaconda.

Then to install, use command the Anaconda prompt to navigate to the project directory and use command `conda env create -f environment.yml`

Then activate the environment with `conda activate caries-model-paper-repo-062026`

----

## Code overview

The model source code is in `src` and uses the agentpy library, which has binaries loaded into src/tools. See src/tools/LICENSE/LICENSE for agentpy licensing details.

Jupyter notebooks for examples of running the model and analyzing results are included at the root directory.

----

## Quick Usage
The first step in running an experiment is to download the code (git clone or the zip available on GitHub) and then install the Python dependencies: 
 - [Jupyter Notebook](https://jupyter.org/install) - for presenting analysis
 - [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) - for organizing data output
 - numpy / [scipy](https://scipy.org/install/) - for computational tools
 - [tqdm](https://pypi.org/project/tqdm/) - for a progress bar when running experiments
 - [joblib](https://joblib.readthedocs.io/en/latest/installing.html) - for parallel model execution when running experiments
 - [Seaborn](https://seaborn.pydata.org/installing.html) / [Matplotlib](https://matplotlib.org/stable/users/installing/index.html) - for data visualization

Next up, you can make a copy of the file 'experiment_templated.ipynb' and run the Jupyter Notebook. Let's run through the different steps in the experiment template.

![Parameters and Imports](/images/template_01.png)

Here, we import the model ['CariesV2']() along with the [Sample](https://agentpy.readthedocs.io/en/stable/reference_sample.html) and [Experiment](https://agentpy.readthedocs.io/en/stable/reference_experiment.html) classes. Below that, we set the default parameters for the model - please do not change these settings in this cell, instead change them in:

![Setting (hyper)parameters](/images/template_02.png)

This cell is where the setup for the experiment occurs. 

### Experiment settings:
 - ``experiment_name``: important if you want to save your results as a .csv file for later access (see section on [data output](#output-details)).
 - ``num_iterations``: since the model is random, it is recommended to run at least 5 iterations per sample (more iterations is ideal, but will quickly increase your run time).
 - ``num_jobs``: the Experiment class implements [joblib](https://joblib.readthedocs.io/en/latest/) to run multiple models in parallel. 
    - *Please* check the number of CPUs available to you and change accordingly 
    - Set to -1 to use all cores (not recommended).

### Some important parameters
First, we copy the parameter dictionary to modify it. Then, we do parameter changes:
```
exp_parameters['init_caries_prevalence'] = 1.00
```
This is an example of setting a new parameter. Here are a couple parameters to keep in mind separate from model behavior; to see explanation of individual behavioral parameters, see the section on the [current model](#current-model)
 - ``'steps'``: the number of time steps which each model will run for.
 - ``'record_steps'``: this parameter controls the time-steps in which our model will record the agent data. Set to 1 to record all time-steps and set to 'steps' to record only at the end.
 - ``'agents'``: this is the number of agents which each model will run with
 - ``'seed'``: this parameter sets the random seed for each model, for reproducibility reasons, but you can remove it or set it to a sample Range.

### Setting samples:
 To set a sample range, set
```
num_steps = <number_of_sample_steps>

range = Range(vmin=<start>, vmax=<end>)
exp_parameters[<parameter_you_wish_to_sample>] = range
```
At the moment, there is no implementation for choosing a step size -- instead, set the appropriate number of sample steps using the equation:
$$\texttt{step\_size}=\frac{\texttt{end}-\texttt{start}}{\texttt{num\_steps}-1}$$
The ''-1'' in the denominator comes from the fact that we index starting at 0, i.e. we are sampling the start and end inclusively. 

The Sample class is set to sample uniformly. In the future, different sampling methods can be added by modifying the Sample class (in fact, the original implementation of [Sample](https://agentpy.readthedocs.io/en/stable/reference_sample.html) has a Saltelli sampling scheme).

![Running the experiment](/images/template_03.png)
This is the cell where the experiment is finally run. If you do not wish to save the data from the experiment, comment out the last line. Data will be saved at root/model_output/ -- after running an experiment, the output will print the exact name of the [DataDict](#output-details) folder.

![Data processing](/images/template_04.png)
Now we load our analytical/graphing tools and then split our data into relevant DataFrames.

``data_filled_missing.variables.Kid``: will give us the DataFrame of all recorded agent variables, including the count of caries/missing/filled/sound for each Kid in the model. This DataFrame uses a [MultiIndex](https://pandas.pydata.org/docs/user_guide/advanced.html) with the indices being $$(\texttt{sample\_id}, \texttt{iteration}, \texttt{obj\_id}, \texttt{t})$$
where 't' is the time-step, 'obj_id' is the id of the individual agents (per model-run), 'iteration' is the iterations number (per sample), and the 'sample_id' is drawn from ``param_samples``, which records what sample each sample_id corresponds to. The image below is an example of how to get the information that you want from the DataFrame:

![Parsing data](/images/template_05.png)

In this example, we are looking at the total number of filled/missing teeth at time t=36. Notice that we get the labels for our graphs by drawing out of the param_samples DataFrame. Here is the outputted heatmap from this code:

![Example heatmap](/images/template_06.png)


----
----

## Current Model

This model is heavily based upon [Agentpy](https://agentpy.readthedocs.io/en/stable/reference.html). For an in-depth of the [ModelFrame](src/tools/ModelFrame.py), [Experiment](src/tools/Experiment.py), and other classes in [src/tools](src/tools/), please check their documentation. Here we will just cover the model which we built on top, the code for which can be found in [CariesKid](/src/CariesKid.py) and [caries_20220319.py](/src/caries_20220319.py) a.k.a. 'CariesV2' in the src/ directory.

The parameters can be broken into a couple groups

----

### Run parameters
 - ``'agents'``: Number of Kid objects created and used for the models run.
 - ``'steps'``: Number of time-steps.
 - ``'record_steps'``: Time between recording steps (see hyper-parameter section above).
 - ``'seed'``: setting the random seed for the models rng. Set to None for complete randomness.

-----

### Behavioral parameters
 - **Decay Parameters**
    - ``'person_decay_prob'``: probability of a Kid receiving at least one cavity at a given time-step
    - ``'sound_to_cavitated'``: tooth-level probability of decay
        - so, 'person_decay_prob' controls the probability of the first sound-to-cavitated tooth in a given time-step, but 'sound_to_cavitated' controls the probability of more teeth decaying after that
    - ``'person_decay_feedback'``: the more decay a Kid currently has (at a given time-step), the more susceptible they are to further decay; this controls the strength of that feedback loop
 - **Filled/Missing Parameters**
    - ``'person_filling_prob'``: probability of person receiving treatment at a time-step
        - treatment will replace all cavitated teeth with filled teeth
        - filled teeth cannot become cavitated
    - ``'cavitated_to_missing'``: tooth-level probability of a cavitated tooth going missing
        - Does not take into account the time spent decayed
 - **Fluoride Parameters**
    - ``'person_preventive_fluoride'``: probability of Kid applying fluoride at each time-step
    - ``'fluoride_effect'``: how effective is fluoride at preventing cavities? If set to 1.00, will prevent all cavities, if set to 0.00, will not prevent any cavities.
    - ``'fluoride_months_effective'``: number of time-steps for which fluoride is effective
    - ``'fluoride_slope'``: how quickly does the effectiveness of fluoride diminish over time
        - *Note*: **Do not change this parameter for now**; needs to be less than 'fluoride_time'
    - ``'fluoride_scenario'``: Controls when/where fluoride is administered. 
        - 1.00: high risk Kids given fluoride at initialization
        - 2.00: fluoride given at time of care
        - 3.00: random (can be any other number)
 - **Initialization Parameters**
    - ``'init_caries_prevalence'``: probability Kid being high-risk at initialization
    - ``'init_caries_prob'``: initial tooth-level decay probability in high-risk Kids
    - ``'init_missing_prob'``: initial tooth-level missing probability in high-risk Kids
        - *Note*: keep 'init_caries_prob'+'init_missing_prob' < 1

----

### What happens at each time-step?
Let's look at the [CariesKid](src/CariesKid.py) code to see what happens at each time-step.

Before the first time-step, ``setup(self)`` occurs. This sets up the list of agents, running the ``setup`` and ``_set_hi_risk`` functions which are on lines 11-35 in [CariesKid.py](src/CariesKid.py). These functions first generate a random number in the range (0,1) and check it against ``'init_caries_prevalence'`` -- this is setting the Kid to high risk or low risk. If the Kid is high risk, they are then given cavities and missing teeth based on ``'init_caries_prob'`` and ``'init_missing_prob'``. If the ``'flouride_scenario'`` is set to 1.00, then the high risk Kids are also given fluoride at this point.

During each time-step, first ``step(self)`` (line 49) is called, and then ``update(self)`` (line 59) is called.
 - ``step(self)`` is broken into three parts, all of which are functions found in [CariesKid.py](src/CariesKid.py):
    - ``Kid._get_treatment()`` (lines 42-67): 
        - generates a random number in (0,1) and fills all cavities if the number is <='person_filling_prob'
        - if treatment does not occur (>'person_filling_prob'), runs through all cavitated teeth and converts them to missing with probability 'cavitated_to_missing'
    - ``Kid._get_cavities()`` (lines 70-101)
        - randomly applies fluoride (see [fluoride parameters](#behavioral-parameters) for details)
        - sets decay probability to 'person_decay_prob' and if the Kid has cavities further increase the decay probability based on 'person_decay_feedback'
        - decreases decay probability based on 'fluoride_effect'
        - if decay occurs, sets one sound tooth to decayed and then decays individual teeth with a probability of 'sound_to_cavitated'
    - ``Kid._fluoride_step()`` (lines 104-114):
        - causes decay in fluoride effectiveness based on 'fluoride_months_effective' and 'fluoride_slope'. Depending on fluoride scenario, will renew fluoride once it is entirely ineffective

Finally, when the model finishes ``end(self)`` is called. Here a set of data points can be calculated or recorded. At the moment, it doesn't do anything, unless 'reporters' are added to our model parameters, something which we will better discuss in the section on [output details](#output-details).
    

## Output Details
When a 'CariesV2' model is run, it will return a [DataDict](src/tools/DataDict.py). See example notebooks for output analysis.








