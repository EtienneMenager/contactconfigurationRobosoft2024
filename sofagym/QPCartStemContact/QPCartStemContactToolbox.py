# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Fab 3 2021"

import numpy as np
from pyquaternion import Quaternion

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib3.animation.animate import Animation

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from common.utils import express_point


SofaRuntime.importPlugin("SofaComponentAll")

class rewardShaper(Sofa.Core.Controller):
    """Compute the reward.

    Methods:
    -------
        __init__: Initialization of all arguments.
        getReward: Compute the reward.
        update: Initialize the value of cost.

    Arguments:
    ---------
        rootNode: <Sofa.Core>
            The scene.

    """
    def __init__(self, *args, **kwargs):
        """Initialization of all arguments.

        Parameters:
        ----------
            kwargs: Dictionary
                Initialization of the arguments.

        Returns:
        -------
            None.

        """
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.rootNode = None
        if kwargs["rootNode"]:
            self.rootNode = kwargs["rootNode"]

    def getReward(self):
        new_value = float(self.rootNode.QPInverseProblemSolver.objective.value)
        reward = abs(self.old_value - new_value)
        self.old_value = new_value
        return reward

    def update(self):
        self.old_value = float(self.rootNode.QPInverseProblemSolver.objective.value)

class goalSetter(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.goalPos = None
        if 'goalPos' in kwargs:
            self.goalPos = kwargs["goalPos"]

    def update(self):
        pass

    def set_mo_pos(self, goal):
        pass


class sceneModerator(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.cartstem=None
        if kwargs["cartstem"]:
            self.cartstem = kwargs["cartstem"]

        self.contacts=None
        if kwargs["contacts"]:
            self.contacts = kwargs["contacts"]


    def getPos(self):
        collis_pos = self.cartstem.cartStem.stemCollis.collisMO.position.value.tolist()
        return [self.cartstem.getPos(), self.contacts.getPos(), collis_pos]

    def setPos(self, all_pos):
        pos, cube_pos, collis_pos = all_pos[0], all_pos[1], all_pos[2]
        self.cartstem.cartStem.MechanicalObject.position.value = np.array(pos)
        self.cartstem.cartStem.MechanicalObject.velocity.value = np.zeros(shape = (21, 6))
        self.cartstem.cartStem.actuator.actuator0.initDisplacement.value = pos[0][0]
        self.cartstem.cartStem.actuator.MechanicalObject.position.value = np.array([pos[0]])
        self.cartstem.cartStem.stemCollis.collisMO.position.value = np.array(collis_pos)
        self.cartstem.cartStem.stemCollis.collisMO.position.value = np.zeros(shape=(148,3))
        self.cartstem.cartStem.actuator.actuator0.init()
        self.contacts.contacts.Cube_1.MechanicalObject.position.value = np.array([[cube_pos[0], 0, 12, 0, 0.382683, 0, 0.923879]])
        self.contacts.contacts.Cube_2.MechanicalObject.position.value = np.array([[cube_pos[1], 0, 12, 0, 0.382683, 0, 0.923879]])

################################################################################

def getState(rootNode):
    actionQP = float(rootNode.cartStem.actuator.MechanicalObject.position.value[0][0])
    max_move = rootNode.sceneModerator.cartstem.max_move

    return [max(min(actionQP, max_move), -max_move)/max_move]

def getReward(rootNode):
    r =  rootNode.Reward.getReward()
    done = r< 0.001
    return done, r

def getPos(root):
    position = root.sceneModerator.getPos()
    return position

def setPos(root, pos):
    root.sceneModerator.setPos(pos)

################################################################################

class applyAction(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root =  kwargs["root"]

    def compute_action(self, actions, nb_step):
        return None

    def apply_action(self, incr):
        pass

def action_to_command(actions, root, nb_step):
    incr = root.applyAction.compute_action(actions, nb_step)
    return incr


def startCmd(root, actions, duration):
    """Initialize the command from root and action.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        action: int
            The action.
        duration: float
            Duration of the animation.

    Returns:
    ------
        None.

    """
    incr = action_to_command(actions, root, duration/root.dt.value + 1)
    startCmd_CartStem(root, incr, duration)


def startCmd_CartStem(rootNode, incr, duration):
    """Initialize the command.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        incr:
            The elements of the commande.
        duration: float
            Duration of the animation.

    Returns:
    -------
        None.
    """

    #Definition of the elements of the animation
    def executeAnimation(rootNode, incr, factor):
        rootNode.applyAction.apply_action(incr)

    #Add animation in the scene
    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"rootNode": rootNode,
                    "incr": incr},
            duration=duration, mode="once"))
