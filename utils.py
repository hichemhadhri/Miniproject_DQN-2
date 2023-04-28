
import numpy as np
import torch
from epidemic_env.env import Observation, ModelDynamics
from constants import *
from collections import namedtuple

class Utils : 
    def action_preprocessor(a:torch.Tensor, dyn:ModelDynamics):
        action = { # DO NOTHING
            'confinement': False, 
            'isolation': False, 
            'hospital': False, 
            'vaccinate': False,
        }
        
        if a == Constants.ACTION_CONFINE:
            action['confinement'] = True
        elif a == Constants.ACTION_ISOLATE:
            action['isolation'] = True
        elif a == Constants.ACTION_VACCINATE:
            action['vaccinate'] = True
        elif a == Constants.ACTION_HOSPITAL:
            action['hospital'] = True
            
        return action
        
    def observation_preprocessor(obs: Observation, dyn:ModelDynamics):
        infected = Constants.SCALE * np.array([np.array(obs.city[c].infected)/obs.pop[c] for c in dyn.cities])
        # compute power 1/4 of the infected
        infected = np.power(infected, 1/4)

        dead = Constants.SCALE * np.array([np.array(obs.city[c].infected)/obs.pop[c] for c in dyn.cities])
        # compute power 1/4 of the dead
        dead = np.power(dead, 1/4)
        
        return torch.Tensor(np.stack((dead, infected))).unsqueeze(0)
    

    Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
