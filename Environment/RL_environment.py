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

class Robinhood_Tutoring(Environment):
    def __init__(self, num_states,num_actions):
        """ Square gridworld RL environment of arbitrary size.
        
        :param size: The number of grid cells on a side 
        :ivar num_states: The number of distinct grid cells
        :ivar env_description: contains attributes describing the environment
        :vartype env_description: :py:class:`.Env_Description`
        :ivar obs: The current obs
        :vartype obs: int
        :ivar terminal_state: Whether the terminal obs is occupied
        :vartype terminal_state: bool
        :ivar time: The current timestep
        :vartype time: int
        :ivar max_time: Maximum allowed timestep
        :vartype max_time: int
        :ivar gamma: The discount factor in calculating the expected return
        :vartype gamma: float
        """
        # self.size = size
        self.num_states = num_states
        self.num_actions = num_actions
        self.env_description = self.create_env_description(self.num_states,self.num_actions)
        self.state = -1
        self.terminal_state = False
        self.time = 0
        self.max_time = 101
        # vis is a flag for visual debugging during obs transitions
        self.vis = False
        self.gamma = 0.9
        self.rews = [[[0.2729,0.1924,0.0358,0.0089,0.0157,0.0537,0.0492,0.0089,0.0179,0.0381,0.3065],
         [0.1751,0.1673,0.0350,0.0039,0.0039,0.0545,0.0233,0.0078,0.0117,0.0856,0.4319],
         [0.3523,0.2911,0.1350,0.0211,0.0148,0.0148,0.0084,0.0105,0.0105,0.0339,0.1076]],
        [[0.3077,0.1795,0.0531,0.0274,0.0201,0.0733,0.0989,0.0128,0.0092,0.0311,0.1869],
         [0.8601,0.1280,0.0089,0.0030],
         [0.4088,0.3205,0.1286,0.0191,0.0115,0.0115,0.0115,0.0058,0.0096,0.0116,0.0615]]]
        self.rew_list = [[[0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9,10]],[[0,1,2,3,4,5,6,7,8,9,10],[0,1,2,10],[0,1,2,3,4,5,6,7,8,9,10]]]

    def create_env_description(self, num_states, num_actions):
        """ Creates the environment description object.  
        :param num_states: The number of states
        :return: Environment description for the obs and action spaces
        :rtype: :py:class:`.Env_Description`
        """
        observation_space = Discrete_Space(0, num_states-1)
        action_space = Discrete_Space(0, num_actions-1)
        return Env_Description(observation_space, action_space)

    def reset(self):
        """ Go back to initial obs and timestep """
        self.state = np.random.randint(2)
        self.time = 0
        self.terminal_state = False

    def get_reward(self,st,action):
        return np.random.choice(self.rew_list[st][action],1,p=self.rews[st][action])[0]

    def transition(self, action):
        """ Transition between states given an action, return a reward. 
        
        :param action: A possible action at the current obs
        :return: reward for reaching the next obs
        """
        reward = 0
        self.time += 1
        # self.update_position(action)

        reward = self.get_reward(self.state,action)
        self.terminal_state  = True

        # if self.is_in_goal_state() or self.time >= self.max_time - 1:
        #     self.terminal_state = True
        #     if self.is_in_goal_state():
        #         reward = 1
        # if self.state == 7:
        #     reward = -1
        if self.vis:
            self.visualize()
            print("reward", reward)
        return reward

    def get_observation(self):
        """ Get the current obs """
        return self.state

    # def update_position(self, action):
    #     """ Helper function for transition() that updates the 
    #     current position given an action 
    #     :param action: A possible action at the current obs
    #     """
    #     if action == 0: #up
    #         if self.state >= self.size: #if not on top row
    #             self.state -= self.size
    #     elif action == 1: #right
    #         if (self.state + 1) % self.size != 0: #not on right column
    #             self.state += 1
    #     elif action == 2: #down
    #         if self.state < self.num_states - self.size: #not on bottom row
    #             self.state += self.size
    #     elif action == 3: #left
    #         if self.state % self.size != 0: #not on left column
    #             self.state -= 1
    #     else:
    #         raise Exception(f"invalid gridworld action {action}")

    # def is_in_goal_state(self):
    #     """ Check whether current obs is goal obs
    #     :return: True if obs is in goal obs, False if not
    #     """
    #     return self.state == self.num_states - 1

    def visualize(self):
        """ Print out current obs information
        """
        # print_state = 0
        # for y in range(self.size):
        #     for x in range(self.size):
        #         if print_state == self.state:
        #             print("A", end="")
        #         else:
        #             print("X", end="")
        #         print_state += 1
        #     print()
        # print()
        print("Gender: ",self.state)