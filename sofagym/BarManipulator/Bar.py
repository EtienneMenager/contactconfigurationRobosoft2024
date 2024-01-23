import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import os
import sys
import numpy as np
import pathlib
import os
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
GeneratedMeshesPath = os.path.dirname(os.path.abspath(__file__))+'/Geometries/'
import Geometries.ConstantsTrunk as Const

class Bar():
    def __init__(self, *args, **kwargs):
        self.collisionGroup = kwargs["collisionGroup"]
        self.translation = kwargs["translation"]
        self.rotation_z = kwargs["rotation_z"]

    def onEnd(self, parent):
        #Add bar to manipulate
        self.bar = parent.addChild('bar')

        self.bar.addObject('EulerImplicitSolver', name='odesolver',rayleighStiffness="0.1", rayleighMass="0.1")
        self.bar.addObject('SparseLDLSolver',name='preconditioner', template="CompressedRowSparseMatrixd")
        self.bar.addObject('GenericConstraintCorrection', solverName='preconditioner')

        self.bar.addObject('MeshVTKLoader', name='loader', filename=GeneratedMeshesPath+'Bar_Volumetric.vtk', translation = [self.translation[0], self.translation[1], -140-self.translation[2]], rotation = [0,0,self.rotation_z])
        self.bar.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
        self.bar.addObject('TetrahedronSetTopologyModifier')
        self.bar.addObject('TetrahedronSetGeometryAlgorithms', template='Vec3d')
        self.bar.addObject('MechanicalObject', name='tetras', rest_position="@loader.position", position="@loader.position", template='Vec3d')
        self.bar.addObject('UniformMass', totalMass=0.2)
        self.bar.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large', poissonRatio=0.3,  youngModulus=5e3)

        self.effectors = self.bar.addChild("effectors")
        pos = [[-40, 0, 0], [0, 0, 0], [40, 0, 0]]
        self.effectors.addObject('MechanicalObject', name='EffectorsMO', position=pos, template='Vec3d', showObject=True, showColor = [1, 0, 1, 1], showObjectScale = 10)
        self.effectors.EffectorsMO.translation.value = [self.translation[0], self.translation[1], -140-self.translation[2]]
        self.effectors.EffectorsMO.rotation.value = [0,0,self.rotation_z]
        self.effectors.addObject("BarycentricMapping")

        #Add bar Visu
        modelVisu = self.bar.addChild("barVisu")
        topo = modelVisu.addObject('MeshSTLLoader', name = "loader", filename=GeneratedMeshesPath+'Bar_Surface.stl')
        modelVisu.addObject('OglModel', name="model", src="@loader", updateNormals=False,  color=[1, 0, 0, 1], rotation = [0,0,self.rotation_z])
        modelVisu.model.translation.value = [self.translation[0], self.translation[1], -140-self.translation[2]]
        modelVisu.addObject('BarycentricMapping')


        #Add bar Collis
        modelCollis = self.bar.addChild('barCollis')
        topo = modelCollis.addObject('MeshSTLLoader', name = "loader", filename=GeneratedMeshesPath+'Bar_Surface.stl')
        modelCollis.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
        modelCollis.addObject('MechanicalObject', name='collisMO', template='Vec3d')
        modelCollis.collisMO.translation.value = [self.translation[0], self.translation[1], -140-self.translation[2]]
        modelCollis.collisMO.rotation.value = [0,0,self.rotation_z]
        modelCollis.addObject('TriangleCollisionModel', group= self.collisionGroup)
        modelCollis.addObject('LineCollisionModel', group=self.collisionGroup)
        modelCollis.addObject('PointCollisionModel', group=self.collisionGroup)
        modelCollis.addObject('BarycentricMapping')

    def getPos(self):
        pos_beam = self.bar.tetras.position.value[:].tolist()
        pos_collis = self.bar.barCollis.collisMO.position.value[:].tolist()
        return [pos_beam, pos_collis]

    def setPos(self, pos):
        pos_beam, pos_collis = pos
        self.bar.tetras.position.value = np.array(pos_beam)
        self.bar.barCollis.collisMO.position.value = np.array(pos_collis)
