#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:30:30 2022

@author: stefan
"""

import numpy as np

# Geometric parameters
NSections = 2
NSegmentsPerSection = 1
NSegments = NSections * NSegmentsPerSection
Length = 60
Height = 10
JointHeight = 6 #< Height
Thickness = 17.5
JointSlopeAngle = np.deg2rad(45)
FixationWidth = 3

# Cavity 2

CavityThickness = 4
CavityWallThickness = 3
CavityLength = Length/2 


OuterRadius = Thickness/2 + 6
NBellowSteps = 1
StepHeight = 5
TeethRadius = Thickness/2
WallThickness = 3
CenterThickness = 1.5
CavityCorkThickness = 3

# Elasticity parameters
PoissonRation = 0.3
YoungsModulus = 3000

# Mold parameters
MoldWallThickness = 3
MoldCoverTolerance = 0.1
LengthMold = 3*Length + 2*MoldWallThickness
LidHoleBorderThickness = 1
LidHoleThickness = Thickness - 2*LidHoleBorderThickness
LidHoleLength = 3*Length/5

MoldHoleThickness = Thickness - 2*LidHoleBorderThickness
MoldHoleLength = Length/2

ThicknessMold = 2*OuterRadius + 2*MoldWallThickness
LengthMold = 3*Length + 2*MoldWallThickness
HeightMold = Height + FixationWidth + MoldWallThickness
MoldHoleLidBorderThickness = 2

#Cable
CableRadius = 0.75
