# -*- coding: utf-8 -*-
"""Create the mesh of the bar.


"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@inria.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 23 2022"

import gmsh
import numpy as np

def init_gmsh(name = "Scene"):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(name)
    gmsh.logger.start()

def _addPoint(X, Y, Z, lc = 3):
    return [gmsh.model.occ.addPoint(XValue, YValue, ZValue, lc) for (XValue,YValue, ZValue) in zip(X,Y,Z)]

def _addLine(src, end, loop = True):
    LineTags = []
    NPoints = len(src)
    for i in range(NPoints):
        LineTags.append(gmsh.model.occ.addLine(src[i], end[i]))

    return LineTags


init_gmsh("Bar")
x_min, x_max = -40, 40
y_min, y_max = -10, 10
z_min, z_max = -10, 10
precision_1, precision_2 = 30, 30

gmsh.model.occ.addBox(x_min, y_min, z_min, x_max-x_min, y_max-y_min, z_max-z_min)

gmsh.model.mesh.field.add("Box", 1)
gmsh.model.mesh.field.setNumber(1, "VIn", precision_1)
gmsh.model.mesh.field.setNumber(1, "VOut", precision_1)
gmsh.model.mesh.field.setNumber(1, "XMin", x_min)
gmsh.model.mesh.field.setNumber(1, "XMax", x_max)
gmsh.model.mesh.field.setNumber(1, "YMin", y_min)
gmsh.model.mesh.field.setNumber(1, "YMax", y_max)
gmsh.model.mesh.field.setNumber(1, "ZMin", z_min)
gmsh.model.mesh.field.setNumber(1, "ZMax", z_max)
gmsh.model.mesh.field.setNumber(1, "Thickness", 0.3)

gmsh.model.mesh.field.setAsBackgroundMesh(1)

gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

#-------------------
# Export
#-------------------

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write("Bar_Volumetric.vtk")

gmsh.model.mesh.clear()

gmsh.model.mesh.field.add("Box", 2)
gmsh.model.mesh.field.setNumber(2, "VIn", precision_2)
gmsh.model.mesh.field.setNumber(2, "VOut", precision_2)
gmsh.model.mesh.field.setNumber(2, "XMin", x_min)
gmsh.model.mesh.field.setNumber(2, "XMax", x_max)
gmsh.model.mesh.field.setNumber(2, "YMin", y_min)
gmsh.model.mesh.field.setNumber(2, "YMax", y_max)
gmsh.model.mesh.field.setNumber(2, "ZMin", z_min)
gmsh.model.mesh.field.setNumber(2, "ZMax", z_max)
gmsh.model.mesh.field.setNumber(2, "Thickness", 0.3)

gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)


gmsh.model.mesh.generate(0)
gmsh.model.mesh.generate(2)
gmsh.write("Bar_Surface.stl")

gmsh.model.occ.synchronize()
gmsh.fltk.run()
