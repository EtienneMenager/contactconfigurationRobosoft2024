# -*- coding: utf-8 -*-
"""Specific environment for the gripper.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Feb 3 2021"

from sofagym.env.common.QPAbstractEnv import AbstractEnv
from sofagym.env.common.QPrpc_server import start_scene
from gym.envs.registration import register

import Sofa
import SofaRuntime
import importlib
import pygame
import glfw
import Sofa.SofaGL
from OpenGL.GL import *
from OpenGL.GLU import *

from gym import spaces
import os
import numpy as np

class QPCartStemContactEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the gripper scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path =  os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "QPCartStemContact",
                      "deterministic": True,
                      "source": [0, -50, 15],
                      "target": [0, 0, 15],
                      "goalList": [[7, 0, 20]],
                      "start_node": None,
                      "scale_factor": 30,
                      "dt": 0.01,
                      "timer_limit": 30,
                      "timeout": 50,
                      "display_size": (1600, 800),
                      "render": 0,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/QPCartStemContact",
                      "planning": False,
                      "discrete": False,
                      "start_from_history": None,
                      "python_version": "python3.9",
                      "zFar": 4000,
                      "time_before_start": 0,
                      "seed": None,
                      "init_x": 5,
                      "cube_x": [-6, 6],
                      "max_move": 7.5,
                      "pos": [[0, 0, i*1.5, 0, -0.707107, 0, 0.707107] for i in range(21)]
                      }


    def __init__(self, config = None):
        super().__init__(config)
        nb_actions = -1
        low = np.array([-1]*1)
        high = np.array([1]*1)
        self.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype='float32')
        self.nb_actions = str(nb_actions)

        dim_state = 1
        low_coordinates = np.array([-1]*dim_state)
        high_coordinates = np.array([1]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates,
                                            dtype='float32')

    def step(self, action):
        return super().step(action)

    def reset(self, config = {"goalList": [[7, 0, 20]], "goalPos": [7, 0, 20], "init_x": 5, "cube_x": [-6, 6], "max_move": 7.5, "pos": [[0, 0, i*1.5, 0, -0.707107, 0, 0.707107] for i in range(21)]}):
        """Reset simulation.

        Note:
        ----
            We launch a client to create the scene. The scene of the program is
            client_<scene>Env.py.

        """
        super().reset()
        self.config.update({'cube_x': config["cube_x"]})
        self.config.update({'init_x': config["init_x"]})
        self.config.update({'goalList': config["goalList"]})
        self.config.update({'max_move': config["max_move"]})
        self.config.update({'goalPos': config["goalPos"]})
        self.config.update({'pos': config["pos"]})

        obs = start_scene(self.config, self.nb_actions)
        return np.array(obs['observation'])

    def get_available_actions(self):
        return self.action_space


register(
    id='qpcartstemcontact-v0',
    entry_point='sofagym.env:QPCartStemContactEnv',
)
