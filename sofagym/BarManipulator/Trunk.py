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
import rigidification

class Trunk():
    def __init__(self, *args, **kwargs):
        self.name = kwargs["name"]
        self.translation = kwargs["translation"]
        self.collisionGroup = kwargs["collisionGroup"]
        self.direct = kwargs["direct"]


    def onEnd(self, parent):
        VolumetricMeshPath = GeneratedMeshesPath + 'Trunk_Volumetric.vtk'
        SurfaceMeshPath = GeneratedMeshesPath + 'Trunk_Surface.stl'

        self.trunk = parent.addChild(self.name)

        self.trunk.addObject('MeshVTKLoader', name='loader', filename=VolumetricMeshPath, scale3d=[1, 1, 1], translation = self.translation)
        self.trunk.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')

        #Rigidification
        BoxMargin = 3
        boxActuator_1 = self.trunk.addObject('BoxROI', name='Actuator_1', box=[self.translation[0]-6, self.translation[1]-6, self.translation[2]-121, self.translation[0]+6, self.translation[1]+6, self.translation[2]-121+10], drawBoxes=False, drawSize = 3, tetrahedra= self.trunk.container.tetrahedra.getLinkPath(), position=self.trunk.container.position.getLinkPath())
        boxActuator_2 = self.trunk.addObject('BoxROI', name='Actuator_2', box=[self.translation[0]-6, self.translation[1]-6, self.translation[2]-61, self.translation[0]+6, self.translation[1]+6, self.translation[2]-61+10], drawBoxes=False, drawSize = 3,tetrahedra=self.trunk.container.tetrahedra.getLinkPath() , position=self.trunk.container.position.getLinkPath())
        boxActuator_3 = self.trunk.addObject('BoxROI', name='Actuator_3', box=[-(Const.Height+2*BoxMargin) + self.translation[0], -(Const.Height+2*BoxMargin) + self.translation[1], BoxMargin + self.translation[2], Const.Height+2*BoxMargin + self.translation[0],Const.Height+2*BoxMargin+ self.translation[1], -BoxMargin + self.translation[2]], drawBoxes=False, drawSize = 3,tetrahedra=self.trunk.container.tetrahedra.getLinkPath() , position=self.trunk.container.position.getLinkPath())

        boxAll = self.trunk.addObject('BoxROI', name='BoxTrunk', box= [self.translation[0]-15, self.translation[1]-15, self.translation[2]+5, self.translation[0]+15, self.translation[1]+15, self.translation[2]-125], drawBoxes=False, drawSize = 3,tetrahedra=self.trunk.container.tetrahedra.getLinkPath() , position=self.trunk.container.position.getLinkPath())

        boxActuator_1.init()
        boxActuator_2.init()
        boxActuator_3.init()
        boxAll.init()

        positionAllPoints = self.trunk.container.findData('position').value;
        nbPoints = len(positionAllPoints)

        indicesActuator_1 = boxActuator_1.indices.value
        indicesActuator_2 = boxActuator_2.indices.value
        indicesActuator_3 = boxActuator_3.indices.value
        indicesDeformable = np.array(list(set(boxAll.indices.value) - set(indicesActuator_1) - set(indicesActuator_2) - set(indicesActuator_3)))

        idxRigid = np.array(indicesActuator_1.tolist() + indicesActuator_2.tolist() + indicesActuator_3.tolist()) #np.append(np.append(indicesActuator_1,indicesActuator_2), indicesActuator_3)
        RigidIndicesTotal = np.sort(idxRigid)
        IdxsOrderedRigid =  np.argsort(idxRigid)
        rigidBlocks = [RigidIndicesTotal.tolist()]

        freeBlocks = np.sort(indicesDeformable).tolist()
        IdxsOrderedFreeBlocks = np.argsort(indicesDeformable)
        indexPairs = np.array(rigidification.fillIndexPairs(nbPoints,freeBlocks,rigidBlocks))

        PointsRigidActuator_1 = np.array(boxActuator_1.pointsInROI.value)
        PointsRigidActuator_2 = np.array(boxActuator_2.pointsInROI.value)
        PointsRigidActuator_3 = np.array(boxActuator_3.pointsInROI.value)

        NPPointsRigid = np.array(PointsRigidActuator_1.tolist()+PointsRigidActuator_2.tolist()+PointsRigidActuator_3.tolist()) #np.append(np.append(PointsRigidActuator_1,PointsRigidActuator_2,0), PointsRigidActuator_3, 0)
        NPSortedPointsRigid = NPPointsRigid[IdxsOrderedRigid,:]
        PointsRigid = NPSortedPointsRigid.tolist()

        PointsDeformable =np.array(boxAll.pointsInROI.value)
        PointsDeformable = np.delete(PointsDeformable, np.array(indicesActuator_1.tolist()+indicesActuator_2.tolist()+indicesActuator_3.tolist()), 0)
        NPSortedPointsDeformable = PointsDeformable[IdxsOrderedFreeBlocks,:]
        PointsDeformable = NPSortedPointsDeformable.tolist()

        NRigidActuator_1 = len(indicesActuator_1)
        NRigidActuator_2 = len(indicesActuator_2)
        NRigidActuator_3 = len(indicesActuator_3)
        rigidIndexPerPointActuator_1 = [0] * (NRigidActuator_1)
        rigidIndexPerPointActuator_2 = [1] * (NRigidActuator_2)
        rigidIndexPerPointActuator_3 = [2] * (NRigidActuator_3)
        NPRigidIndexPerPoint = np.array(rigidIndexPerPointActuator_1 + rigidIndexPerPointActuator_2 + rigidIndexPerPointActuator_3)
        NPSortedRigidIndexPerPoint = NPRigidIndexPerPoint[IdxsOrderedRigid]
        rigidIndexPerPoint = NPSortedRigidIndexPerPoint.flatten().tolist()

        solverNode = self.trunk.addChild("solverNode")
        solverNode.addObject('EulerImplicitSolver', name='odesolver',rayleighStiffness="0.1", rayleighMass="0.1")
        solverNode.addObject('SparseLDLSolver',name='preconditioner', template="CompressedRowSparseMatrixd")
        solverNode.addObject('GenericConstraintCorrection', solverName='preconditioner')
        solverNode.addObject('MechanicalMatrixMapper', name="MMM", template='Vec3d,Rigid3d', object1='@./deformableNode/DeformableMech', object2='@./RigidNode/RigidMesh', nodeToParse='@./deformableNode/model' )

        deformableNode = solverNode.addChild("deformableNode")
        deformableNode.addObject('PointSetTopologyContainer', position=PointsDeformable)
        deformableNode.addObject('MechanicalObject', name='DeformableMech')


        self.RigidNode = solverNode.addChild('RigidNode')
        RigidOrientation = [0, 0, 0, 1]
        self.RigidNode.addObject("MechanicalObject",template="Rigid3d",name="RigidMesh", position=[[self.translation[0], self.translation[1], self.translation[2]-120 + 5]+RigidOrientation, [self.translation[0], self.translation[1], self.translation[2]-60 + 5]+RigidOrientation, [self.translation[0], self.translation[1], self.translation[2]- 5]+RigidOrientation], showObject=True, showObjectScale=5)
        RigidifiedNode = self.RigidNode.addChild('RigidifiedNode')
        RigidifiedNode.addObject('MechanicalObject', name='RigidifiedMesh', position = PointsRigid, template = 'Vec3d')
        RigidifiedNode.addObject("RigidMapping",globalToLocalCoords="true", rigidIndexPerPoint=rigidIndexPerPoint)

        model = deformableNode.addChild('model')
        RigidifiedNode.addChild(model)
        model.addObject('EulerImplicitSolver', name='odesolver')
        model.addObject('ShewchukPCGLinearSolver', iterations='15', name='linearsolver', tolerance='1e-5', update_step='1')
        model.addObject('MeshVTKLoader', name='loader', filename=VolumetricMeshPath, translation = self.translation)
        model.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
        model.addObject('TetrahedronSetGeometryAlgorithms')
        model.addObject('MechanicalObject', name='tetras', template='Vec3', showIndices=False, showIndicesScale='4e-5')
        model.addObject('UniformMass', totalMass='0.1')

        extremityBox = model.addObject('BoxROI', name='extremity', box=[self.translation[0]-11, self.translation[1]-11, self.translation[2]-121, self.translation[0]+11, self.translation[1]+11, self.translation[2]-121+30], drawBoxes=False, drawSize = 3,tetrahedra=self.trunk.container.tetrahedra.getLinkPath() , position=self.trunk.container.position.getLinkPath())
        extremityBox.init()
        indicesExtremity = extremityBox.indices.value
        youngModulus = [500 if i in indicesExtremity else Const.YoungsModulus for i in range(280)]
        model.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=Const.PoissonRation,  youngModulus= youngModulus)
        model.addObject("SubsetMultiMapping",name="subsetMapping",template="Vec3d,Vec3d", input='@'+deformableNode.getPathName()+'/DeformableMech' + ' ' + '@'+RigidifiedNode.getPathName()+'/RigidifiedMesh' , output='@./tetras', indexPairs=indexPairs.tolist())

        ##########################################
        # Visualization                          #
        ##########################################

        modelVisu = model.addChild('visu')
        modelVisu.addObject('MeshSTLLoader', filename=SurfaceMeshPath, name="loader", translation=self.translation)
        modelVisu.addObject('OglModel', src="@loader", scale3d=[1, 1, 1])
        modelVisu.addObject('BarycentricMapping')

        ##########################################
        # Collision                              #
        ##########################################

        modelCollis = model.addChild('collis')
        modelCollis.addObject('MeshSTLLoader', filename=SurfaceMeshPath, name="loader", translation=self.translation)
        modelCollis.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
        modelCollis.addObject('MechanicalObject', name='collisMO', template='Vec3d')
        modelCollis.addObject('TriangleCollisionModel', group=self.collisionGroup)
        modelCollis.addObject('LineCollisionModel', group=self.collisionGroup)
        modelCollis.addObject('PointCollisionModel', group=self.collisionGroup)
        modelCollis.addObject('BarycentricMapping')

        ##########################################
        # Cavity                              #
        ##########################################

        CavitySurfaceMeshPath = GeneratedMeshesPath+'Cavity.stl'
        self.cavity = model.addChild('Cavity')
        self.cavity.addObject('MeshSTLLoader', name='MeshLoader', filename=CavitySurfaceMeshPath, translation = self.translation)
        self.cavity.addObject('MeshTopology', name='topology', src='@MeshLoader')
        self.cavity.addObject('MechanicalObject', src="@topology")
        self.cavity.addObject('SurfacePressureConstraint', template='Vec3d', triangles='@topology.triangles')
        self.cavity.addObject('BarycentricMapping', name="Mapping", mapForces="false", mapMasses="false")

        ##########################################
        # Effectors                              #
        ##########################################
        nb_points = 3
        scale = 10
        effector_pos = [[self.translation[0], self.translation[1], self.translation[2]-120*(1-i/(nb_points+scale))] for i in range(nb_points)]
        self.effector = model.addChild('Effectors')
        self.effector.addObject('MechanicalObject', name="EffectorMO", showObject=True, showColor = [1, 0, 0, 1], showObjectScale = 10, position = effector_pos)
        self.effector.addObject('BarycentricMapping', mapForces=False, mapMasses=False)

        ##########################################
        # Sensors                              #
        ##########################################

        height = -1.90*Const.Length
        sensorsAngle = np.deg2rad(90)
        radius = Const.Height
        sensor_pos = [[np.cos(i*sensorsAngle)*radius + self.translation[0], np.sin(sensorsAngle*i)*radius + self.translation[1], height + self.translation[2]] for i in range(4)]

        self.sensors = model.addChild('Sensors')
        self.sensors.addObject('MechanicalObject', name="SensorMO", showObject=True, showColor = [0, 0, 1, 1], showObjectScale = 10, position = sensor_pos)
        self.sensors.addObject('BarycentricMapping', mapForces=False, mapMasses=False)

        return self.set_actuators()

    def set_actuators(self):
        if self.direct:
            self.control = self.trunk.addChild("control")
            controlMO_1 = self.control.addObject("MechanicalObject", name = "controlMO_1", template = 'Rigid3', position= [self.translation[0], self.translation[1], self.translation[2]-120 + 5, 0, 0, 0, 1])
            controlMO_2 = self.control.addObject("MechanicalObject", name = "controlMO_2", template = 'Rigid3', position= [self.translation[0], self.translation[1], self.translation[2]-60 + 5, 0, 0, 0, 1])
            controlMO_3 = self.control.addObject("MechanicalObject", name = "controlMO_3", template = 'Rigid3', position= [self.translation[0], self.translation[1], self.translation[2] - 5, 0, 0, 0, 1])
            self.RigidNode.addObject('RestShapeSpringsForceField', name='restSprint_1', points=0, external_rest_shape=controlMO_1.getLinkPath(), stiffness=1, angularStiffness=1e12)
            self.RigidNode.addObject('RestShapeSpringsForceField', name='restSprint_2', points=1, external_rest_shape=controlMO_2.getLinkPath(), stiffness=1, angularStiffness=1e12)
            self.RigidNode.addObject('RestShapeSpringsForceField', name='restSprint_3', points=2, external_rest_shape=controlMO_3.getLinkPath(), stiffness=1e12, angularStiffness=1e12)
            return [controlMO_1, controlMO_2, controlMO_3]

    def getPos(self):
        pos_trunk = self.trunk.solverNode.deformableNode.model.tetras.position.value[:].tolist()
        pos_collis = self.trunk.solverNode.deformableNode.model.collis.collisMO.position.value[:].tolist()
        pos_deformable = self.trunk.solverNode.deformableNode.DeformableMech.position.value[:].tolist()
        pos_rigidNode = self.trunk.solverNode.RigidNode.RigidMesh.position.value[:].tolist()
        pos_rigidifiedNode = self.trunk.solverNode.RigidNode.RigidifiedNode.RigidifiedMesh.position.value[:].tolist()
        pos_cavity = self.cavity.MechanicalObject.position.value[:].tolist()
        pos_sensor = self.sensors.SensorMO.position.value[:].tolist()

        pos_effector = self.effector.EffectorMO.position.value[:].tolist()

        pos_control_1 = self.control.controlMO_1.position.value[:].tolist()
        pos_control_2 = self.control.controlMO_2.position.value[:].tolist()
        return [pos_trunk, pos_collis, pos_deformable, pos_rigidNode, pos_rigidifiedNode,pos_effector, pos_control_1, pos_control_2, pos_cavity, pos_sensor]

    def setPos(self, pos):
        pos_trunk, pos_collis, pos_deformable, pos_rigidNode, pos_rigidifiedNode,pos_effector, pos_control_1, pos_control_2, pos_cavity, pos_sensor= pos
        self.trunk.solverNode.deformableNode.model.tetras.position.value = np.array(pos_trunk)
        self.trunk.solverNode.deformableNode.model.collis.collisMO.position.value = np.array(pos_collis)

        self.trunk.solverNode.deformableNode.DeformableMech.position = np.array(pos_deformable)
        self.trunk.solverNode.RigidNode.RigidMesh.position = np.array(pos_rigidNode)
        self.trunk.solverNode.RigidNode.RigidifiedNode.RigidifiedMesh.position = np.array(pos_rigidifiedNode)
        self.effector.EffectorMO.position = np.array(pos_effector)
        self.sensors.SensorMO.position.value = np.array(pos_sensor)
        self.cavity.MechanicalObject.position.value = np.array(pos_cavity)
        self.control.controlMO_1.position.value = np.array(pos_control_1)
        self.control.controlMO_2.position.value = np.array(pos_control_2)
