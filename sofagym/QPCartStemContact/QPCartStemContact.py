# -*- coding: utf-8 -*-
"""Create the Cartstem


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"

import os
import numpy as np

import sys
import importlib
import pathlib
import json

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
pathMesh = os.path.dirname(os.path.abspath(__file__))+'/mesh/'

from common.utils import addRigidObject

class CartStem():
    def __init__(self, *args, **kwargs):

        if "cartstem_config" in kwargs:
            print(">>  Init QPcartstem_config...")
            self.cartstem_config = kwargs["cartstem_config"]

            self.init_pos = self.cartstem_config["init_pos"]
            self.cart_size = self.cartstem_config["cart_size"]
            self.max_move =  self.cartstem_config["max_move"]
            self.max_v =  self.cartstem_config["max_v"] #cm/s
            self.dt =  self.cartstem_config["dt"]
            self.beam_config =  self.cartstem_config["beam_config"]
            self.use_direct_action = self.cartstem_config["use_direct_action"]

            self.pos = kwargs["pos"]

        else:
            print(">>  No cartstem_config ...")
            exit(1)


    def onEnd(self, rootNode, collisionGroup=1):
        print(">>  Init QPCartStem...")

        #Add Stem
        self.cartStem = self._create_beam(rootNode)

        #Add visual cart
        cartVisu = self.cartStem.addChild("cartVisu")
        cartVisu.addObject('MeshOBJLoader', name="loader", filename=pathMesh+"cube.obj")
        cartVisu.addObject('OglModel', name="model", src="@loader", scale3d=self.cart_size, color=[1, 0, 0, 1], updateNormals=False)
        cartVisu.addObject('RigidMapping', index = 0)

        #Add stem Visu
        stemVisu = self.cartStem.addChild("stemVisu")
        stemVisu.addObject('MeshSTLLoader', name="loader", filename=pathMesh+"obstacleVisu.stl")
        stemVisu.addObject('OglModel', name="model", src="@loader", scale3d=[0.01, 0.25, 0.01], color=[1, 0, 0, 1], updateNormals=False, rotation = [-90, 0, 0], translation = [self.init_pos[0], -2.5, 0])
        stemVisu.addObject('SkinningMapping')

        #Add stem collision
        stemCollis = self.cartStem.addChild('stemCollis')
        stemCollis.addObject('MeshSTLLoader', name='loader', filename=pathMesh+"obstacleColli.stl", scale3d=[0.01, 0.25, 0.01], rotation = [-90, 0, 0], translation = [self.init_pos[0], -2.5, 0])
        stemCollis.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
        stemCollis.addObject('MechanicalObject', name='collisMO', template='Vec3d')
        stemCollis.addObject('TriangleCollisionModel', group=collisionGroup)
        stemCollis.addObject('LineCollisionModel', group=collisionGroup)
        stemCollis.addObject('PointCollisionModel', group=collisionGroup)
        stemCollis.addObject('SkinningMapping')

        #Add path Visu
        pathVisu = rootNode.addChild("pathVisu")
        pathVisu.addObject('MeshSTLLoader', name="loader", filename=pathMesh+"obstacleVisu.stl")
        pathVisu.addObject('OglModel', name="model", src="@loader", scale3d=[0.005, 0.5, 0.005], color=[0, 0, 0, 1], updateNormals=False, translation = [30, 0, -1], rotation = [0, 0, -90])


    def _create_beam(self, rootNode):
        model = rootNode.addChild('cartStem')
        model.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
        model.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixd")

        init_pos = self.beam_config['init_pos']
        tot_length = self.beam_config['tot_length']
        nb_frame = self.beam_config['nbFrames']

        edges = [[i, i+1] for i in range(nb_frame)]
        model.addObject('EdgeSetTopologyContainer', position=self.pos, edges=edges)
        model.addObject('MechanicalObject', template='Rigid3',
                            position= self.pos,
                            rest_position = [[0, 0, i*(tot_length/nb_frame), 0, -0.707107, 0, 0.707107] for i in range(nb_frame+1)],
                            showObject=False, drawMode=0, showObjectScale=0.5)

        interpolation = model.addObject('BeamInterpolation', straight=False, defaultYoungModulus=1e6, radius=0.5)
        model.addObject('AdaptiveBeamForceFieldAndMass', computeMass=True, massDensity=1e-5, interpolation = interpolation.getLinkPath())
        model.addObject('PartialFixedConstraint', indices=0, fixedDirections=[0, 1, 1, 1, 1, 1])
        model.addObject('GenericConstraintCorrection')

        if self.use_direct_action:
            self.goal = rootNode.addChild("rest_shape")
            goalMO = self.goal.addObject("MechanicalObject", name = "goalMO", template = 'Rigid3', position= [init_pos[0], 0, 0, 0, 0, 0, 1])
            model.addObject('RestShapeSpringsForceField', name='control', points=0,
                external_rest_shape=goalMO.getLinkPath(), stiffness=1e3)

        return model

    def getPos(self):
        pos_stem = self.cartStem.MechanicalObject.position.value.tolist()
        return pos_stem

    def setPos(self, pos):
        pos_stem = pos
        self.cartStem.MechanicalObject.position.value = np.array(pos_stem)

class Contacts():
    def __init__(self, *args, **kwargs):

        if "contact_config" in kwargs:
            print(">>  Init QPcontact_config...")
            self.contact_config = kwargs["contact_config"]

            self.init_pos = self.contact_config["init_pos"]
            self.cube_size = self.contact_config["cube_size"]
            self.cube_x = self.contact_config["cube_x"]
        else:
            print(">>  No contact_config ...")
            exit(1)

    def onEnd(self, rootNode):
        print(">>  Init QPContact..")
        self.contacts = rootNode.addChild('contacts')

        #ADD Cube1
        pos_Cube_1 = [p for p in self.init_pos]
        pos_Cube_1[0] = pos_Cube_1[0] + self.cube_x[0]
        self.Cube_1 = addRigidObject(self.contacts, filename=pathMesh+'cube.obj',name='Cube_1',scale= self.cube_size, position= pos_Cube_1 + [ 0, 0.3826834, 0, 0.9238795])
        self.Cube_1.addObject('FixedConstraint', indices=0)

        #ADD Cube2
        pos_Cube_2 = [p for p in self.init_pos]
        pos_Cube_2[0] = pos_Cube_2[0] + self.cube_x[1]
        self.Cube_2 = addRigidObject(self.contacts, filename=pathMesh+'cube.obj',name='Cube_2',scale= self.cube_size, position= pos_Cube_2 + [ 0, 0.3826834, 0, 0.9238795])
        self.Cube_2.addObject('FixedConstraint', indices=0)

    def getPos(self):
        return self.cube_x + self.cube_size
