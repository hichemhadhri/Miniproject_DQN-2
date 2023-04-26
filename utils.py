
import numpy as np
import torch
from epidemic_env.env import Observation, ModelDynamics
from constants import *

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
        dead = Constants.SCALE * np.array([np.array(obs.city[c].infected)/obs.pop[c] for c in dyn.cities])
        confined = np.ones_like(dead)*int((dyn.get_action()['confinement']))
        return torch.Tensor(np.stack((infected, dead, confined))).unsqueeze(0)