from seldonian.RL.environments.Environment import *
from seldonian.RL.Env_Description.Env_Description import *
from seldonian.RL.RL_runner import run_trial
from seldonian.utils.io_utils import save_pickle,load_pickle
from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
from seldonian.RL.Env_Description.Env_Description import Env_Description
from seldonian.RL.Env_Description.Spaces import Discrete_Space
from seldonian.spec import createRLSpec
from seldonian.dataset import RLDataSet
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.plot_utils import plot_gradient_descent
from seldonian.utils.io_utils import load_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.RL.RL_runner import create_agent,run_trial_given_agent_and_env

from experiments.generate_plots import RLPlotGenerator
from Environment.RL_environment import Robinhood_Tutoring

import autograd.numpy as np   # Thinly-wrapped version of Numpy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from statistics import mean, variance
from tqdm import tqdm


trial_dict = {}
trial_dict["env"] = Robinhood_Tutoring(2,3)
trial_dict["agent"] = "Parameterized_non_learning_softmax_agent"
trial_dict["num_episodes"] = 10000
trial_dict["num_trials"] = 1
trial_dict["vis"] = False


# Generate the contextual bandit episodes
episodes, agent = run_trial(trial_dict)
for ep in episodes:
    print(ep)
episodes_file = './robinhood_tutoring_1000episodes.pkl'
save_pickle(episodes_file,episodes)


episodes_file = './robinhood_tutoring_1000episodes.pkl'
episodes = load_pickle(episodes_file)
dataset = RLDataSet(episodes=episodes)


# Initialize the spec file
observation_space = Discrete_Space(0, 1)
action_space = Discrete_Space(0, 2)
env_description =  Env_Description(observation_space, action_space)
policy = DiscreteSoftmax(hyperparam_and_setting_dict={},
    env_description=env_description)
env_kwargs={'gamma':1}
save_dir = '.'
constraint_strs = ['J_pi_new >= 2.9743']
deltas=[0.05]

spec = createRLSpec(
    dataset=dataset,
    policy=policy,
    constraint_strs=constraint_strs,
    deltas=deltas,
    env_kwargs=env_kwargs,
    save=True,
    save_dir='.',
    verbose=True)


# load specfile
specfile = './spec.pkl'
spec = load_pickle(specfile)
spec.optimization_hyperparams['num_iters']=100
spec.optimization_hyperparams['alpha_theta']=0.05
spec.optimization_hyperparams['alpha_lamb']=0.01

# Run Seldonian algorithm 
SA = SeldonianAlgorithm(spec)
passed_safety,solution = SA.run(write_cs_logfile=True)
if passed_safety:
    print("Passed safety test!")
    print("The solution found is:")
    print(solution)
else:
    print("No Solution Found")
print("Primary objective (-IS estimate) evaluated on safety dataset:")
print(SA.evaluate_primary_objective(branch='safety_test',theta=solution))


# Generate Gradient Descent Plots
cs_file = './logs/candidate_selection_log0.p'
solution_dict = load_pickle(cs_file)
fig = plot_gradient_descent(solution_dict,
    primary_objective_name='- IS estimate',
    save=False)
plt.show()