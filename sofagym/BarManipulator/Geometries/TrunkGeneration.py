#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 12:10:07 2022

@author: stefan
"""

import gmsh
import numpy as np
import locale
import ConstantsTrunk as Constants
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

def createLines(PointTags):

    LineTags = np.empty((0,1),dtype=int)
    NPoints = len(PointTags)
    for i in range(1, NPoints):
        LineTags = np.append(LineTags, [gmsh.model.occ.addLine(PointTags[i-1], PointTags[i])])

    LineTags = np.append(LineTags, [gmsh.model.occ.addLine(PointTags[-1], PointTags[0])])

    #print('LineTags: ' + str(LineTags))
    return LineTags

def makeSegmentTipMod(SegmentDimTag, Length, Height, JointHeight, Thickness, JointSlopeAngle, FixationWidth=3, lc=1):

    Point1Tag = gmsh.model.occ.addPoint(-Thickness/2,JointHeight,-Length)

    LengthDiagonal = (Height-JointHeight)/np.cos(JointSlopeAngle)
    JointStandoff = LengthDiagonal*np.sin(JointSlopeAngle)
    Point2Tag = gmsh.model.occ.addPoint(-Thickness/2,Height,JointStandoff-Length)
    CenterPointTag = gmsh.model.occ.addPoint(-Thickness/2,0,-Length/2)
    CircleArcTag = gmsh.model.occ.addCircleArc(Point1Tag, CenterPointTag, Point2Tag)

    pass

def makeSegmentStage1Mod(SegmentDimTag, Length, Height, JointHeight, Thickness, JointSlopeAngle, lc=1):

    LengthDiagonal = (Height-JointHeight)/np.cos(JointSlopeAngle)
    JointStandoff = LengthDiagonal*np.sin(JointSlopeAngle)
    Stage1DistanceToStandoff = 2
    Stage1DistanceFromBase = 5
    Overboarding = 3
    SubstractionDepth = 2.5
    ZOffset = JointStandoff+Stage1DistanceToStandoff
    BoxSubstractionOuterDimTag = (3,gmsh.model.occ.addBox(-Thickness/2-Overboarding,
                                                          Stage1DistanceFromBase,
                                                          -ZOffset,
                                                          Thickness+2*Overboarding,
                                                          Height+Overboarding,
                                                          -(Length-2*ZOffset)))

    BoxSubstractionInnerDimTag = (3,gmsh.model.occ.addBox(-Thickness/2+SubstractionDepth,
                                                          Stage1DistanceFromBase,
                                                          -ZOffset,
                                                          Thickness-2*SubstractionDepth,
                                                          Height-(SubstractionDepth+Stage1DistanceFromBase),
                                                          -(Length-2*ZOffset)))

    CutOut = gmsh.model.occ.cut([BoxSubstractionOuterDimTag], [BoxSubstractionInnerDimTag])
    SubstractionObjectDimTag = CutOut[0][0]

    CutOut = gmsh.model.occ.cut([SegmentDimTag],[SubstractionObjectDimTag])
    Segment1ModDimTag = CutOut[0][0]
    return Segment1ModDimTag

def makeSegmentFixationMod(SegmentDimTag, Length, Height, JointHeight, Thickness, JointSlopeAngle, FixationWidth, lc=1):


    CylinderFillDimTag = (3,gmsh.model.occ.addCylinder(0,0,0, 0,0,-Length/2, Height,angle=np.pi))
    CylinderFixationDimTag = (3,gmsh.model.occ.addCylinder(0,0,0, 0,0,-FixationWidth, Height+FixationWidth,angle=np.pi))
    gmsh.model.occ.rotate([CylinderFillDimTag,CylinderFixationDimTag],0,0,0,0,0,1, np.pi/2)
    gmsh.model.occ.synchronize()
    FuseOut = gmsh.model.occ.fuse([SegmentDimTag],[CylinderFillDimTag,CylinderFixationDimTag])
    SegmentDimTag = FuseOut[0][0]
    return SegmentDimTag

def createSegment(Length, Height, JointHeight, Thickness, JointSlopeAngle,lc=1):

    PointTags = np.empty((0,1), int)

    LengthDiagonal = (Height-JointHeight)/np.cos(JointSlopeAngle)
    JointStandoff = LengthDiagonal*np.sin(JointSlopeAngle)

    YValues = np.array([0,0,JointHeight,Height,Height,JointHeight])
    ZValues = np.array([0,-Length,-Length, -Length+JointStandoff,-JointStandoff,0])

    PointTags = np.append(PointTags, [gmsh.model.occ.addPoint(0, YValue, ZValue, lc) for (YValue,ZValue) in zip(YValues,ZValues)])

    LineTags = createLines(PointTags)
    WireLoop = gmsh.model.occ.addWire(LineTags)
    SurfaceTag = gmsh.model.occ.addPlaneSurface([WireLoop])
    gmsh.model.occ.synchronize()    
    RevolveTags = gmsh.model.occ.revolve([(2,SurfaceTag)],0,0,0,0,0,1,np.pi)
    print("Segment extrude dim tags:", RevolveTags)
    gmsh.model.occ.synchronize()    
    return RevolveTags[1]

def createArticulationBellow(OuterRadius, NBellowSteps, StepHeight, TeethRadius, FingerWidth, lc=1):

    PointTags = np.empty((0,1), int)

    TotalHeight = NBellowSteps * StepHeight
    ZValues = np.linspace(0, TotalHeight, NBellowSteps+1)
    NPoints = len(ZValues)
    XValues = np.ones((NPoints,))
    XValues[0::2] = OuterRadius
    XValues[1::2] = TeethRadius

    PointTags = np.append(PointTags, [gmsh.model.occ.addPoint(0,0,0, lc)])
    PointTags = np.append(PointTags, [gmsh.model.occ.addPoint(XValue, 0, ZValue, lc) for (XValue,ZValue) in zip(XValues,ZValues)])
    PointTags = np.append(PointTags, [gmsh.model.occ.addPoint(0,0,TotalHeight, lc)])

    LineTags = createLines(PointTags)
    WireLoop = gmsh.model.occ.addWire(LineTags)
    SurfaceTag = gmsh.model.occ.addPlaneSurface([WireLoop])

    RevolveDimTags = gmsh.model.occ.revolve([(2,SurfaceTag)], 0,0,0, 0,0,1, np.pi)
    HalfDimTag = RevolveDimTags[1]

    HalfCopyDimTag = gmsh.model.occ.copy([HalfDimTag])
    gmsh.model.occ.affineTransform(HalfCopyDimTag, [1,0,0,0, 0,1,0,0, 0,0,-1,0])

    FusionOut = gmsh.model.occ.fuse([HalfDimTag], HalfCopyDimTag)
    BellowDimTag = FusionOut[0]

    return BellowDimTag

def createCavity2(CavityLength, CavityThickness, CavityWallThickness, Length, Height, JointHeight, Thickness, JointSlopeAngle,lc=1):

    PointTags = np.empty((0,1), int)
    
    YOffset = Height-(CavityThickness+CavityWallThickness)    
    LengthDiagonal = (Height-JointHeight)/np.cos(JointSlopeAngle)
    JointStandoff = LengthDiagonal*np.sin(JointSlopeAngle)
    CavityLengthOffset = JointStandoff    
    
    YValues = np.array([YOffset, YOffset+CavityThickness, YOffset+CavityThickness, YOffset])    
    ZValues = np.array([-Length+CavityWallThickness+CavityLength, -Length+CavityWallThickness+CavityLength, -Length+CavityWallThickness+JointStandoff, -Length+CavityWallThickness])    

    PointTags = np.append(PointTags, [gmsh.model.occ.addPoint(0, YValue, ZValue, lc) for (YValue,ZValue) in zip(YValues,ZValues)])

    LineTags = createLines(PointTags)
    WireLoop = gmsh.model.occ.addWire(LineTags)
    SurfaceTag = gmsh.model.occ.addPlaneSurface([WireLoop])
    gmsh.model.occ.synchronize()    
    RevolveTags = gmsh.model.occ.revolve([(2,SurfaceTag)],0,0,0,0,0,1,np.pi)
    print("Segment revolve:", RevolveTags)
    
    HalfCopyDimTag = gmsh.model.occ.copy([RevolveTags[1]])
    print("HalfCopyDimTag: ", HalfCopyDimTag)
    gmsh.model.occ.affineTransform(HalfCopyDimTag, [-1,0,0,0, 0,1,0,0, 0,0,1,0])    
    FusionOut = gmsh.model.occ.fuse([RevolveTags[1]], HalfCopyDimTag)
    CavityBaseDimTags = FusionOut[0]    
    return CavityBaseDimTags 


def createCavitySketch(OuterRadius, NBellowSteps, StepHeight, TeethRadius, WallThickness, CenterThickness, lc=1):

    PointTags = np.empty((0,1), int)

    TotalHeight = NBellowSteps * StepHeight
    ZValues = np.linspace(0, TotalHeight, NBellowSteps+1)
    NPoints = len(ZValues)
    XValues = np.ones((NPoints,))
    XValues[0::2] = OuterRadius - WallThickness
    XValues[1::2] = TeethRadius - WallThickness

    PointTags = np.append(PointTags, [gmsh.model.occ.addPoint(0, 0, 0, lc)])
    PointTags = np.append(PointTags, [gmsh.model.occ.addPoint(XValue, 0, ZValue, lc) for (XValue,ZValue) in zip(XValues,ZValues)])
    PointTags = np.append(PointTags, [gmsh.model.occ.addPoint(0, 0, TotalHeight, lc)])

    LineTags = createLines(PointTags)
    WireLoop = gmsh.model.occ.addWire(LineTags)
    SurfaceTag = gmsh.model.occ.addPlaneSurface([WireLoop])
    return SurfaceTag    

def createCavityVolume(OuterRadius, NBellowSteps, StepHeight, TeethRadius, WallThickness, CenterThickness, CavityCorkThickness, lc=1):

    TotalHeight = NBellowSteps * StepHeight
    SurfaceTag = createCavitySketch(OuterRadius, NBellowSteps, StepHeight, TeethRadius, WallThickness,CenterThickness)
    RevolveDimTags = gmsh.model.occ.revolve([(2,SurfaceTag)], 0,0,0, 0,0,1, np.pi/2)
    HalfDimTag = RevolveDimTags[1]

    HalfCopyDimTag = gmsh.model.occ.copy([HalfDimTag])
    print("HalfCopyDimTag: ", HalfCopyDimTag)
    gmsh.model.occ.affineTransform(HalfCopyDimTag, [1,0,0,0, 0,1,0,0, 0,0,-1,0])

    FusionOut = gmsh.model.occ.fuse([HalfDimTag], HalfCopyDimTag)

    CavityBaseDimTag = FusionOut[0]

    Box1Tag = gmsh.model.occ.addBox(-CenterThickness,
                                    0,
                                    -(TotalHeight+1),
                                    2*CenterThickness,
                                    OuterRadius,
                                    2*(TotalHeight+1))

    Box2Tag = gmsh.model.occ.addBox(-OuterRadius,
                                    0,
                                    -(TotalHeight+1),
                                    2*OuterRadius,
                                    CavityCorkThickness,
                                    2*(TotalHeight+1))

    BoxDimTags = [(3,Box1Tag),(3,Box2Tag)]

    CutOut = gmsh.model.occ.cut(CavityBaseDimTag,BoxDimTags)
    print("CutOut: ", CutOut)
    CavityDimTags = CutOut[0]
    return CavityDimTags

def createAllCavities(OuterRadius, NBellowSteps, StepHeight, TeethRadius, WallThickness, CenterThickness,lc=1):

    Cavity1DimTags = createCavityVolume(Constants.OuterRadius, Constants.NBellowSteps, Constants.StepHeight, Constants.TeethRadius, Constants.WallThickness, Constants.CenterThickness, Constants.CavityCorkThickness, lc=lc)
    gmsh.model.occ.translate(Cavity1DimTags,0,0,-Constants.Length)
    Cavity2DimTags = gmsh.model.occ.copy(Cavity1DimTags)
    Cavity3DimTags = gmsh.model.occ.copy(Cavity1DimTags)
    Cavity4DimTags = gmsh.model.occ.copy(Cavity1DimTags)

    gmsh.model.occ.affineTransform(Cavity2DimTags, [-1,0,0,0, 0,1,0,0, 0,0,1,0])
    gmsh.model.occ.translate(Cavity3DimTags,0,0,-Constants.Length)
    gmsh.model.occ.affineTransform(Cavity4DimTags, [-1,0,0,0, 0,1,0,0, 0,0,1,0])
    gmsh.model.occ.translate(Cavity4DimTags,0,0,-Constants.Length)
    gmsh.model.occ.synchronize()
    AllCavitiesDimTags = Cavity1DimTags + Cavity2DimTags + Cavity3DimTags + Cavity4DimTags
    return AllCavitiesDimTags

def exportCavity2(lc=4):    
    gmsh.model.add("Cavity")
    CavityDimTags = createCavity2(Constants.CavityLength, Constants.CavityThickness, Constants.CavityWallThickness, Constants.Length, Constants.Height, Constants.JointHeight, Constants.Thickness, Constants.JointSlopeAngle,lc=1)    
    gmsh.model.occ.translate(CavityDimTags, 0,0,-Constants.Length)
    gmsh.model.occ.synchronize()
    
    
    gmsh.model.mesh.field.add("Box", 6)
    gmsh.model.mesh.field.setNumber(6, "VIn", lc)
    gmsh.model.mesh.field.setNumber(6, "VOut", lc)
    gmsh.model.mesh.field.setNumber(6, "XMin", -Constants.Thickness)
    gmsh.model.mesh.field.setNumber(6, "XMax", Constants.Thickness)
    gmsh.model.mesh.field.setNumber(6, "YMin", 0)
    gmsh.model.mesh.field.setNumber(6, "YMax", Constants.Height)
    gmsh.model.mesh.field.setNumber(6, "ZMin", -Constants.NSegments*Constants.Length)
    gmsh.model.mesh.field.setNumber(6, "ZMax", 0)
    gmsh.model.mesh.field.setNumber(6, "Thickness", 0.3)

    gmsh.model.mesh.field.setAsBackgroundMesh(6)

    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

    
    gmsh.model.mesh.generate(2)
    gmsh.write("Cavity.stl")
    gmsh.clear()
def exportCavities(lc=5):
    #-------------------
    # Segments
    #-------------------
    gmsh.model.add("Cavity")

    #-------------------
    # Cavities
    #-------------------

    Cavity1DimTags = createCavityVolume(Constants.OuterRadius, Constants.NBellowSteps, Constants.StepHeight, Constants.TeethRadius, Constants.WallThickness, Constants.CenterThickness, Constants.CavityCorkThickness, lc=lc)
    gmsh.model.occ.translate(Cavity1DimTags,0,0,-Constants.Length)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write("Cavity01.stl")

    gmsh.clear()

    Cavity1DimTags = createCavityVolume(Constants.OuterRadius, Constants.NBellowSteps, Constants.StepHeight, Constants.TeethRadius, Constants.WallThickness, Constants.CenterThickness, Constants.CavityCorkThickness, lc=lc)
    gmsh.model.occ.translate(Cavity1DimTags,0,0,-Constants.Length)
    gmsh.model.occ.affineTransform(Cavity1DimTags, [-1,0,0,0, 0,1,0,0, 0,0,1,0])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write("Cavity02.stl")

    gmsh.clear()

    Cavity1DimTags = createCavityVolume(Constants.OuterRadius, Constants.NBellowSteps, Constants.StepHeight, Constants.TeethRadius, Constants.WallThickness, Constants.CenterThickness, Constants.CavityCorkThickness, lc=lc)
    gmsh.model.occ.translate(Cavity1DimTags,0,0,-2*Constants.Length)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write("Cavity03.stl")

    gmsh.clear()

    Cavity1DimTags = createCavityVolume(Constants.OuterRadius, Constants.NBellowSteps, Constants.StepHeight, Constants.TeethRadius, Constants.WallThickness, Constants.CenterThickness, Constants.CavityCorkThickness, lc=lc)
    gmsh.model.occ.translate(Cavity1DimTags,0,0,-2*Constants.Length)
    gmsh.model.occ.affineTransform(Cavity1DimTags, [-1,0,0,0, 0,1,0,0, 0,0,1,0])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write("Cavity04.stl")

    gmsh.clear()

def createTrunk(lc = 7):
    #-------------------
    # Segments
    #-------------------

    

    Segment1DimTag = createSegment(Constants.Length, Constants.Height, Constants.JointHeight, Constants.Thickness, Constants.JointSlopeAngle)    
    SegmentsDimTags = []
    for i in range(1,Constants.NSegments):
        NewSegmentDimTags = gmsh.model.occ.copy([Segment1DimTag])
        SegmentsDimTags.append(NewSegmentDimTags[0])
        gmsh.model.occ.translate(NewSegmentDimTags,0,0,i*-Constants.Length)

    Segment1DimTag = makeSegmentFixationMod(Segment1DimTag, Constants.Length, Constants.Height, Constants.JointHeight, Constants.Thickness, Constants.JointSlopeAngle, Constants.FixationWidth)

    print("SegmentsDimTags: {}".format(SegmentsDimTags))
    FuseOut = gmsh.model.occ.fuse([Segment1DimTag],SegmentsDimTags)
    TrunkHalfDimTag = FuseOut[0][0]
    TrunkHalfCopyDimTags = gmsh.model.occ.copy([TrunkHalfDimTag])
    gmsh.model.occ.affineTransform(TrunkHalfCopyDimTags,[-1,0,0,0, 0,1,0,0, 0,0,1,0])
    FuseOut = gmsh.model.occ.fuse([TrunkHalfDimTag],TrunkHalfCopyDimTags)
    TrunkDimTag = FuseOut[0][0]


    CavityDimTags = createCavity2(Constants.CavityLength, Constants.CavityThickness, Constants.CavityWallThickness, Constants.Length, Constants.Height, Constants.JointHeight, Constants.Thickness, Constants.JointSlopeAngle,lc=1)    
    gmsh.model.occ.translate(CavityDimTags, 0,0,-Constants.Length)
    
    
    CutOut = gmsh.model.occ.cut([TrunkDimTag], CavityDimTags)
    gmsh.model.occ.synchronize()
    
    
    TrunkWithCavityDimTag = CutOut[0]
    
#    #-------------------
#    # Bellows
#    #-------------------
#
#    Bellow1DimTag = createArticulationBellow(Constants.OuterRadius, Constants.NBellowSteps, Constants.StepHeight, Constants.TeethRadius, Constants.Thickness, lc=lc)
#    Bellow2DimTags = gmsh.model.occ.copy(Bellow1DimTag)
#
#    gmsh.model.occ.translate(Bellow1DimTag,0,0,-Constants.Length)
#    gmsh.model.occ.translate(Bellow2DimTags,0,0,-2*Constants.Length)
#
#    FuseOut = gmsh.model.occ.fuse(Bellow1DimTag,[Segment1DimTag]+Segment2DimTags + Segment3DimTags + Bellow2DimTags)
#    FingerNoCavitiesDimTag = FuseOut[0]
#    gmsh.model.occ.synchronize()
#
#    #-------------------
#    # Cavities
#    #-------------------
#
#    AllCavitiesDimTags = createAllCavities(Constants.OuterRadius, Constants.NBellowSteps, Constants.StepHeight, Constants.TeethRadius, Constants.WallThickness, Constants.CenterThickness, lc=lc)
#
#
#    print("AllCavitiesDimTags: ", AllCavitiesDimTags)
#    #-------------------
#    # Cut Cavities
#    #-------------------
#
#    CutOut = gmsh.model.occ.cut(FingerNoCavitiesDimTag,AllCavitiesDimTags)
#    FingerDimTag = CutOut[0][0]
#    gmsh.model.occ.synchronize()
#
    #-------------------
    # MeshSizes
    #-------------------

    gmsh.model.mesh.field.add("Box", 6)
    gmsh.model.mesh.field.setNumber(6, "VIn", lc)
    gmsh.model.mesh.field.setNumber(6, "VOut", lc)
    gmsh.model.mesh.field.setNumber(6, "XMin", -Constants.Thickness)
    gmsh.model.mesh.field.setNumber(6, "XMax", Constants.Thickness)
    gmsh.model.mesh.field.setNumber(6, "YMin", 0)
    gmsh.model.mesh.field.setNumber(6, "YMax", Constants.Height)
    gmsh.model.mesh.field.setNumber(6, "ZMin", -Constants.NSegments*Constants.Length)
    gmsh.model.mesh.field.setNumber(6, "ZMax", 0)
    gmsh.model.mesh.field.setNumber(6, "Thickness", 0.3)

    gmsh.model.mesh.field.setAsBackgroundMesh(6)

    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

    #-------------------
    # Export
    #-------------------

    gmsh.model.occ.synchronize()
    gmsh.write("Trunk_Parametric.step")
    gmsh.model.mesh.generate(3)
    gmsh.write("Trunk_Volumetric.vtk")

    gmsh.model.mesh.clear()
    #gmsh.model.mesh.generate(0)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.refine()
    gmsh.write("Trunk_Surface.stl")


    return TrunkDimTag


def createShapes():
    exportCavity2(lc=2)
    FingerDimTag = createTrunk()    
    gmsh.fltk.run()    

if __name__=="__main__":
    createShapes()
