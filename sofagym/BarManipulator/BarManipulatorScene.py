# -*- coding: utf-8 -*-
"""Create the scene with the Abstraction of Jimmy.


Units: mm, kg, s.
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
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from common.header import addHeader as header
from common.header import addVisu as visu
from common.utils import addRigidObject

from Trunk import Trunk
from Bar import Bar
from Controller import Controller, StartingPointController
from BarManipulatorToolbox import rewardShaper, goalSetter, sceneModerator, applyAction

import os
GeneratedMeshesPath = os.path.dirname(os.path.abspath(__file__))+'/Geometries/'


def add_goal_node(root, orientation):
    goal = root.addChild("Goal")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3, showColor=[1, 0, 0, 0.25], position= [0, 0, -142.502])
    modelVisu = goal.addChild("Goal_visu")
    topo = modelVisu.addObject('MeshSTLLoader', name = "loader", filename=GeneratedMeshesPath+'Bar_Surface.stl')
    modelVisu.addObject('OglModel', name="model", src="@loader", updateNormals=False,  color=[1, 0, 0, 0.2], rotation = [0,0,-orientation])
    modelVisu.model.translation.value = [0, 0, -142.502]
    modelVisu.addObject('BarycentricMapping')


def create_init_action(rotation_z):
    init_action = []

    init_action += [[0, 0, 0.7, 0, 0, 0, -0.7, 0, -1] for i in range(3)]
    init_action += [[0, 0, 0.7, 0, 0, 0, -0.7, 0, 0.7] for i in range(5)]
    init_action += [[0, 0, 0.3, 0, 0, 0, -0.3, 0, 0.7] for i in range(5)]

    return init_action

def createScene(rootNode, config = {"source": [0, -370, -50],
                                    "target": [0, 0, -75],
                                    "goalPos": 45,
                                    "seed": None,
                                    "zFar":4000,
                                    "dt": 0.01,
                                    "case": 2,
                                    "time_before_start": 13,
                                    "translation": [0, 0, 0],
                                    "rotation_z": 0,
                                    "data_collection": False},
                         mode = 'simu_and_visu'):


    header(rootNode, alarmDistance=5.0, contactDistance=1.5, tolerance = 1e-6, maxIterations=500, gravity = [0,0,-9810.0], dt = config['dt'], mu = "0.7", angleCone=0.2, coneFactor = 0)
    rootNode.addObject('RequiredPlugin', name='SofaMiscMapping')
    #angleCone = 0.1 et coneFactor = 0
    position_spot = [[0, -500, 100]]
    direction_spot = [[0.0, 1, 0]]
    visu(rootNode, config, position_spot, direction_spot, cutoff = 250)

    trunk_0 = Trunk(name = "Trunk_0", translation = [20, 0, 0], collisionGroup = 0, direct = True)
    trunk_1 = Trunk(name = "Trunk_1", translation = [-20, 0, 0], collisionGroup = 1, direct = True)
    trunks = [trunk_0, trunk_1]

    actuators_0 = trunk_0.onEnd(rootNode)
    actuators_1 = trunk_1.onEnd(rootNode)
    actuators = [actuators_0, actuators_1]


    #Add visual base
    baseVisu = trunk_0.trunk.solverNode.deformableNode.model.addChild("BaseVisu")
    baseVisu.addObject('MeshOBJLoader', name="loader", filename=GeneratedMeshesPath+"cube.obj")
    baseVisu.addObject('OglModel', name="model", src="@loader", scale3d=[40, 20, 2], color=[0.5, 0.5, 0.5, 1], updateNormals=False)
    baseVisu.addObject("BarycentricMapping")

    bar = Bar(collisionGroup = 3, translation = config["translation"], rotation_z = config["rotation_z"])
    bar.onEnd(rootNode)


    floor = addRigidObject(rootNode,filename=GeneratedMeshesPath+"cube.obj",name='Floor',scale=[100, 100,1], position=[0,0,-155,  0, 0, 0, 1], collisionGroup=1)
    floor.addObject('FixedConstraint', indices=0)

    
    add_goal_node(rootNode, config["goalPos"])
    rootNode.addObject(goalSetter(name="GoalSetter", goalPos=config["goalPos"]))
    rootNode.addObject(rewardShaper(name="Reward", root = rootNode, verbose = False, init_rot = abs(config["rotation_z"])))

    rootNode.addObject(sceneModerator(name="sceneModerator",  bar = bar, trunks = trunks, actuators = actuators))


    # init_action = create_init_action(config["rotation_z"])
    # assert len(init_action)== config["time_before_start"]
    init_action = []
    rootNode.addObject(applyAction(name="applyAction", actuators= actuators, max_incr_rot=45, max_orientation = 45, max_incr_trans = 40, max_translation = 40, init_action = init_action))

    if config["data_collection"]:
        rootNode.addObject(StartingPointController(name="Controller", root = rootNode, case = config["case"], bar_orientation = config["rotation_z"]))
    # rootNode.addObject(Controller(name="Controller", root = rootNode, actuators = actuators))
