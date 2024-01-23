# -*- coding: utf-8 -*-
"""Create the scene with the Abstraction of Jimmy.


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"

VISUALISATION = False

import sys
import importlib
import pathlib
import Sofa
import numpy as np
import json

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from common.header import addHeader as header
from common.header import addVisu as visu
from common.utils import addRigidObject

from QPCartStemContact import CartStem, Contacts
from QPCartStemContactToolbox import sceneModerator, goalSetter, rewardShaper, applyAction

def add_goal_node(root, pos):
    goal = root.addChild("Goal")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=False)
    goal.addObject('MeshOBJLoader', name = "loader", filename='mesh/cylinder.obj', scale3d=[0.05, 3, 0.05], rotation = [90, 0, 0], translation = [pos[0], pos[1], pos[2]-20])
    goal.addObject('OglModel', src='@loader',color=[1, 0, 0, 0.5])
    return goal_mo

class MoveGoal(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root =  kwargs["root"]
        self.init_pos = True

    def onAnimateBeginEvent(self, event):
        current_effector_goals = self.root.cartStem.Effectors.EffectorMO.position.value.tolist()
        current_pos = self.root.Goals.GoalMO.position.value.tolist()
        current_pos[0][2] = current_effector_goals[0][2]
        self.root.Goals.GoalMO.position = current_pos



def createScene(rootNode, config = {"source": [0, -50, 10],
                                    "target": [0, 0, 10],
                                    "goalPos": [7, 0, 20],
                                    "seed": None,
                                    "zFar":4000,
                                    "init_x": 0,
                                    "cube_x": [-6, 6],
                                    "max_move": 7.5,
                                    "dt": 0.01,
                                    "pos": [[0, 0, i*1.5, 0, -0.707107, 0, 0.707107] for i in range(21)]},
                         mode = 'simu_and_visu', pos = None):


    header(rootNode, alarmDistance=1.0, contactDistance=0.1, tolerance = 1e-6, maxIterations=100, gravity = [0,0,-981.0], dt = config['dt'], genericConstraintSolver = False, mu = "0")

    rootNode.addObject('RequiredPlugin', name='SoftRobots.Inverse')
    if "simu" in mode:
        rootNode.addObject('QPInverseProblemSolver')


    position_spot = [[0, -50, 10]]
    direction_spot = [[0.0, 1, 0]]
    visu(rootNode, config, position_spot, direction_spot, cutoff = 250)


    max_move =  config['max_move']
    assert config['cube_x'][0] < config['cube_x'][1]
    bound = [config['cube_x'][0]+3, config['cube_x'][1]-3]
    init_x = max(-min(config["init_x"], bound[1]), bound[0])
    pos = config["pos"]
    if pos == []:
        pos = [[init_x, 0, i*1.5, 0, -0.707107, 0, 0.707107] for i in range(21)]


    max_v = 2
    beam_config = {'init_pos': [0, 0, 0], 'tot_length': 30, 'nbFrames': 20}
    cartstem_config = {"init_pos": [0, 0, 0], "cart_size": [2, 2, 5], "max_move": max_move,  "max_v": max_v, "dt": config["dt"],  "beam_config":beam_config, "use_direct_action":False}
    contact_config = {"init_pos": [0, 0, 12], "cube_size": [2, 1, 2], "cube_x": config["cube_x"]}

    contacts = Contacts(contact_config = contact_config)
    contacts.onEnd(rootNode)

    collisionGroup=1
    cartstem = CartStem(cartstem_config = cartstem_config, pos = pos, collisionGroup=collisionGroup)
    cartstem.onEnd(rootNode)

    #Goals
    init_pos = [config["goalPos"][0], config["goalPos"][1], beam_config["tot_length"]]
    goal = rootNode.addChild('Goals')
    GoalMO = goal.addObject('MechanicalObject', name='GoalMO', position=init_pos)
    goal.addObject('SphereCollisionModel', radius='1.0', group=collisionGroup, color="red")
    rootNode.addObject(MoveGoal(name="MoveGoal", root = rootNode))

    #Effectors
    effector = cartstem.cartStem.addChild('Effectors')
    effector.addObject('MechanicalObject', name="EffectorMO", showObject=True, translation = [0, 0, 0])
    effector.addObject('PositionEffector', template='Vec3', indices = [0], effectorGoal= GoalMO.position.getLinkPath(), useDirections=[1, 0, 0, 0, 0, 0])
    effector.addObject('SphereCollisionModel', radius='1.0', group=collisionGroup, color="green")
    effector.addObject('RigidMapping', index = beam_config['nbFrames'])

    #Actuators
    actuator = cartstem.cartStem.addChild('actuator')
    actuator.addObject('MechanicalObject', template = 'Rigid3', position = [pos[0][0], 0, 0, 0, 0, 0, 1])
    actuator.addObject('SlidingActuator', template='Rigid3', name="actuator0", direction = [1, 0, 0, 0, 0, 0],
                            maxNegativeDisp=max_move, maxPositiveDisp=max_move, maxDispVariation=max_v*config["dt"], indices = 0,
                            initDisplacement  = pos[0][0])
    actuator.addObject('RigidRigidMapping')


    add_goal_node(rootNode, config["goalPos"])
    rootNode.addObject(goalSetter(name="GoalSetter", goalPos=config["goalPos"]))
    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, max_dist= cartstem_config['max_move']))
    rootNode.addObject(applyAction(name="applyAction", root= rootNode))
    rootNode.addObject(sceneModerator(name="sceneModerator",  cartstem = cartstem, contacts = contacts))
