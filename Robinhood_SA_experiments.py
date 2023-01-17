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
import sys 
  

import autograd.numpy as np   # Thinly-wrapped version of Numpy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from statistics import mean, variance
from tqdm import tqdm

def generate_episodes_and_calc_J(**kwargs):
        """ Calculate the expected discounted return 
        by generating episodes

        :return: episodes, J, where episodes is the list
            of generated ground truth episodes and J is
            the expected discounted return
        :rtype: (List(Episode),float)
        """
        # Get trained model weights from running the Seldonian algo
        model = kwargs['model']
        new_params = model.policy.get_params()
       
        # create env and agent
        hyperparameter_and_setting_dict = kwargs['hyperparameter_and_setting_dict']
        agent = create_agent(hyperparameter_and_setting_dict)
        env = Robinhood_Tutoring(2,3)
       
        # set agent's weights to the trained model weights
        agent.set_new_params(new_params)
        
        # generate episodes
        num_episodes = kwargs['n_episodes_for_eval']
        episodes = run_trial_given_agent_and_env(
            agent=agent,env=env,num_episodes=num_episodes)

        # Calculate J, the discounted sum of rewards
        returns = np.array([weighted_sum_gamma(ep.rewards,env.gamma) for ep in episodes])
        J = np.mean(returns)
        return episodes,J

def Robinhood_experiments(spec_file):
    # Parameter setup
    run_experiments = True
    make_plots = True
    save_plot = False
    performance_metric = 'J(pi_new)'
    n_trials = 20

    # We generate data fractions as a log space
    data_fracs = np.logspace(-2.3,0,10)

    n_workers = 8
    verbose=True
    results_dir = f'results/gridworld_{n_trials}_trials'
    os.makedirs(results_dir,exist_ok=True)
    plot_savename = os.path.join(results_dir,f'gridworld_{n_trials}trials.png')
    n_episodes_for_eval = 1000

    # Load spec
    specfile = spec_file
    spec = load_pickle(specfile)
    spec.optimization_hyperparams['num_iters'] = 40
    spec.optimization_hyperparams['alpha_theta'] = 0.05
    spec.optimization_hyperparams['alpha_lamb'] = 0.01
    spec.optimization_hyperparams['beta_velocity'] = 0.9
    spec.optimization_hyperparams['beta_rmspropr'] = 0.95

    perf_eval_fn = generate_episodes_and_calc_J
    perf_eval_kwargs = {'n_episodes_for_eval':n_episodes_for_eval}

    hyperparameter_and_setting_dict = {}
    hyperparameter_and_setting_dict["env"] = Robinhood_Tutoring(2,3)
    hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    hyperparameter_and_setting_dict["num_episodes"] = 5000
    hyperparameter_and_setting_dict["num_trials"] = 1
    hyperparameter_and_setting_dict["vis"] = False

    plot_generator = RLPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        n_workers=n_workers,
        datagen_method='generate_episodes',
        hyperparameter_and_setting_dict=hyperparameter_and_setting_dict,
        perf_eval_fn=perf_eval_fn,
        perf_eval_kwargs=perf_eval_kwargs,
        results_dir=results_dir,
        )
    if run_experiments:
        plot_generator.run_seldonian_experiment(verbose=verbose)

    if make_plots:
        if save_plot:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label=performance_metric,
                savename=plot_savename)
        else:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label=performance_metric,)

if __name__ == "__main__":
    spec_file = sys.argv[1]
    Robinhood_experiments(spec_file)