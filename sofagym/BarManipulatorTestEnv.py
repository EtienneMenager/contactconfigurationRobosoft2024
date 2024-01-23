# -*- coding: utf-8 -*-
"""Specific environment for the trunk (simplified).
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 7 2020"

from sofagym.env.common.AbstractTestEnv import AbstractEnv
from sofagym.env.common.rpc_server_test import start_scene, get_infos
from gym.envs.registration import register

from gym import spaces
import os
import numpy as np


class BarManipulatorTestEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the trunk scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "BarManipulator",
                      "deterministic": True,
                      "goalList": [0],
                      "source": [0, -370, -50],
                      "target": [0, 0, -75],
                      "start_node": None,
                      "scale_factor": 10,
                      "timer_limit": 30,
                      "timeout": 200,
                      "dt": 0.01,
                      "display_size": (1600, 800),
                      "render": 1,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/BarManipulatorTest",
                      "planning": False,
                      "discrete": False,
                      "seed": None,
                      "start_from_history": None,
                      "python_version": "python3.9",
                      "zFar":4000,
                      "time_before_start": 0,
                      "translation": [0, 0, 0],
                      "rotation_z": 0,
                      "data_collection": False}


    def __init__(self, config=None):
        super().__init__(config)

        nb_actions = -1
        low_coordinates = np.array([-1]*9)
        high_coordinates = np.array([1]*9)
        self.action_space = spaces.Box(low_coordinates, high_coordinates,
                                           dtype='float32')
        self.nb_actions = str(nb_actions)

        dim_state = 23
        low_coordinates = np.array([-5]*dim_state)
        high_coordinates = np.array([5]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates,
                                            dtype='float32')

    def step(self, action):
        return super().step(action)

    def reset(self, goal = None, translation = [0, 0, 0], init_rotation_z = 0):
        """Reset simulation.

        Note:
        ----
            We launch a client to create the scene. The scene of the program is
            client_<scene>Env.py.

        """
        super().reset()
        if init_rotation_z is None:
            rotation_z = np.random.randint(0, 159)
        else:
            rotation_z = init_rotation_z

        if goal is None:
            self.goal = np.random.randint(rotation_z+1, 360)
            #self.goal = np.random.choice([np.random.randint(rotation_z+1, 360) , np.random.randint(-360, -1)], p = [0.5, 0.5])
            # self.goal = np.random.choice([np.random.randint(rotation_z+1, 270) , np.random.randint(-270, -1)], p = [0.5, 0.5])
        else:
            self.goal = goal


        self.config.update({'goalPos': self.goal})
        self.config.update({"translation": translation, "rotation_z": -rotation_z})
        obs = start_scene(self.config, self.nb_actions)

        return np.array(obs['observation'])

    def get_available_actions(self):
        """Gives the actions available in the environment.

        Parameters:
        ----------
            None.

        Returns:
        -------
            list of the action available in the environment.
        """
        return list(range(int(self.nb_actions)))

    def get_infos(self):
        infos = get_infos(self.past_actions)['infos']
        return infos


register(
    id='barmanipulatortest-v0',
    entry_point='sofagym.env:BarManipulatorTestEnv',
)
