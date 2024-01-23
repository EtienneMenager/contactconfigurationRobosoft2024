# -*- coding: utf-8 -*-
"""Contx_roller for the Abstraction of Jimmy.


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "March 8 2021"

import Sofa
import json
import numpy as np
from pyquaternion import Quaternion
from operator import itemgetter
import itertools

class Controller(Sofa.Core.Controller):

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.actuators = kwargs['actuators']

        self.max_incr_rot = 10
        self.max_incr_trans = 5
        self.max_orientation = 45
        self.max_translation = 30
        self.a_rotation, self.b_rotation =  self.max_orientation, 0
        self.a_translation, self.b_translation = self.max_translation/2,  self.max_translation/2

        self.n_trunk = [0, 0, 0, 0, 1, 1, 1, 1]
        self.n_part =  [0, 0, 1, 1, 0, 0, 1, 1]
        self.n_angle = [0, 1, 0, 1, 0, 1, 0, 1]

        self.current_part = 0
        self.current_trunk = 0

        self.current_ang = np.zeros(shape = (2, 2, 3))
        for i in range(2):
            self.current_ang[i][0][2] = 180
            self.current_ang[i][1][2] = 180

        self.root.Reward.update()
        print(">>  Controler init.")

    def _compute_distances(self, n_trunk, indices, points_trunk, bar_center, bar_point_G, normal_bar, v_bar):
        if indices != set():
            collis_trunk = self.root.sceneModerator.trunks[n_trunk].trunk.solverNode.deformableNode.model.collis.collisMO.position.value
            point_indices = list(itertools.chain.from_iterable(itemgetter(*indices)(points_trunk)))
            bary_collis_trunk = collis_trunk[point_indices,:].mean(axis=0)
            v_contact_trunk_center, v_contact_trunk_G = (bary_collis_trunk-bar_center), (bary_collis_trunk-bar_point_G)
            bar_side = np.dot(v_contact_trunk_center, normal_bar)
            dist_xy, dist_z = abs(bar_side), abs(np.dot(v_contact_trunk_center, np.array([0,0,1])))
            along_bar = np.dot(v_contact_trunk_G, v_bar)
        else:
            bar_side, dist_xy, dist_z, along_bar = None, None, None, None
        return bar_side, dist_xy, dist_z, along_bar

    def _compute_ms(self, distances_0, distances_1, print_info = False, tresh_xy = [5, 15], tresh_z = 10.5):
        bar_side_0, dist_xy_0, dist_z_0, along_bar_0 = distances_0
        bar_side_1, dist_xy_1, dist_z_1, along_bar_1 = distances_1
        ms = None

        if dist_xy_0 is not None and dist_xy_1 is not None and tresh_xy[0] < dist_xy_0 < tresh_xy[1] and tresh_xy[0] < dist_xy_1 < tresh_xy[1] and dist_z_0<tresh_z and dist_z_1<tresh_z :
            if bar_side_0 >= 0 and bar_side_1 < 0:
                if along_bar_0 <= along_bar_1:
                    ms = 1
                else:
                    ms = 2
            elif bar_side_0 < 0 and bar_side_1 >= 0:
                if along_bar_0 <= along_bar_1:
                    ms = 3
                else:
                    ms = 4
            else:
                ms = 0
        else:
            ms = 0

        if print_info:
            print("\n[INFO] >> dist (xy):", [dist_xy_0, dist_xy_1])
            print("[INFO] >> dist (z):", [dist_z_0, dist_z_1])
            print("[INFO] >> bar_side:", [bar_side_0, bar_side_1])
            print("[INFO] >> along_bar:", [along_bar_0, along_bar_1])
            print("[INFO] >> Meta-states:", ms)

        return ms

    def onAnimateBeginEvent(self, eventType):
        barCollis = self.root.sceneModerator.bar.bar.barCollis.collisMO.constraint.value
        trunkCollis_0 = self.root.sceneModerator.trunks[0].trunk.solverNode.deformableNode.model.collis.collisMO.constraint.value
        trunkCollis_1 = self.root.sceneModerator.trunks[1].trunk.solverNode.deformableNode.model.collis.collisMO.constraint.value

        idx_bar, points_bar = self._dealConstraints(barCollis)
        idx_trunk_0, points_trunk_0 =  self._dealConstraints(trunkCollis_0)
        idx_trunk_1, points_trunk_1 =  self._dealConstraints(trunkCollis_1)

        bar_point_A= self.root.sceneModerator.bar.effectors.EffectorsMO.position.value[0, :3]
        bar_point_B = self.root.sceneModerator.bar.effectors.EffectorsMO.position.value[2, :3]
        bar_center = self.root.sceneModerator.bar.effectors.EffectorsMO.position.value[1, :3]

        if bar_point_A[0] <= bar_point_B[0]:
            bar_point_G, bar_point_D = bar_point_A, bar_point_B
        else:
            bar_point_D, bar_point_G = bar_point_A, bar_point_B

        v_bar = (bar_point_D-bar_point_G)/np.linalg.norm(bar_point_D-bar_point_G)
        normal_bar = np.cross(np.array([0, 0, 1]), v_bar)

        indices_bar_trunk_0 = idx_bar.intersection(idx_trunk_0)
        indices_bar_trunk_1 = idx_bar.intersection(idx_trunk_1)

        bar_side_0, dist_xy_0, dist_z_0, along_bar_0 = self._compute_distances(0, indices_bar_trunk_0, points_trunk_0, bar_center, bar_point_G, normal_bar, v_bar)
        bar_side_1, dist_xy_1, dist_z_1, along_bar_1= self._compute_distances(1, indices_bar_trunk_1, points_trunk_1, bar_center, bar_point_G, normal_bar, v_bar)
        self._compute_ms([bar_side_0, dist_xy_0, dist_z_0, along_bar_0],
                          [bar_side_1, dist_xy_1, dist_z_1, along_bar_1], print_info = False)

        print("[INFO]  >> REWARD:", self.root.Reward.getReward())

    def _dealConstraints(self, s, particular_point = None):
        inter_s = s.split("\n")
        idx = set()
        correspondance = {}
        for constraint in inter_s:
            cons = self._dealConstraint(constraint)
            if particular_point is not None and cons['id'] is not None:
                if particular_point in cons["points"]:
                    id = cons['id']
                else:
                    id = None
            else:
                id = cons['id']
            if id is not None:
                idx.add(id)
                correspondance.update({id: cons["points"]})
        return idx, correspondance

    def _dealConstraint(self, s):
        if s!='':
            inter_s = s.split(' ')

            num_constraint = int(inter_s[0])
            num_points = int(inter_s[1])

            points = []
            constraint_point = []
            for i in range(num_points):
                point = int(inter_s[2+5*i])
                coord_1 = float(inter_s[3+5*i])
                coord_2 = float(inter_s[4+5*i])
                coord_3 = float(inter_s[5+5*i])
                points.append(point)
                constraint_point.append([coord_1, coord_2, coord_3])


            return {'id': num_constraint, 'nb_point': num_points, 'points': points, 'constraint': constraint_point}
        else:
            return {'id': None}

    def euler_2_quat(self, z_yaw=np.pi/2, y_pitch=0.0, x_roll=np.pi):
        z_yaw = np.pi - z_yaw
        z_yaw_matrix = np.array([[np.cos(z_yaw), -np.sin(z_yaw), 0.0],[np.sin(z_yaw), np.cos(z_yaw), 0.0], [0, 0, 1.0]])
        y_pitch_matrix = np.array([[np.cos(y_pitch), 0., np.sin(y_pitch)], [0.0, 1.0, 0.0], [-np.sin(y_pitch), 0, np.cos(y_pitch)]])
        x_roll_matrix = np.array([[1.0, 0, 0], [0, np.cos(x_roll), -np.sin(x_roll)], [0, np.sin(x_roll), np.cos(x_roll)]])
        rot_mat = z_yaw_matrix.dot(y_pitch_matrix.dot(x_roll_matrix))
        return Quaternion(matrix=rot_mat).elements

    def _changeTranslation(self, incr):
        pos =  min(0, max(-self.max_translation, self.actuators[0][2].position.value[0][2]+incr))
        with self.actuators[0][2].position.writeable() as pos_0:
            pos_0[0][2] = pos
        with self.actuators[1][2].position.writeable() as pos_1:
            pos_1[0][2] = pos


    def _changeOrientation(self, n_part, n_trunk):
        if n_part == 1:
            ang = np.radians(self.current_ang[n_trunk][1])
            new_q = self.euler_2_quat(z_yaw=ang[2], y_pitch=ang[1], x_roll=ang[0]).tolist()
            new_q = new_q[1:]+[new_q[0]]
            pos = self.actuators[n_trunk][1].position.value[0][:3].tolist()
            self.actuators[n_trunk][1].position.value = [pos + new_q]
        ang = np.array([p for p in self.current_ang[n_trunk][0]])
        ang[:2]+= self.current_ang[n_trunk][1][:2]
        ang = np.radians(ang)
        new_q = self.euler_2_quat(z_yaw=ang[2], y_pitch=ang[1], x_roll=ang[0]).tolist()
        new_q = new_q[1:]+[new_q[0]]
        pos = self.actuators[n_trunk][0].position.value[0][:3].tolist()
        self.actuators[n_trunk][0].position.value = [pos + new_q]

    def _modifyAngle(self, n_part, n_trunk, n_angle, incr):
        self.current_ang[n_trunk][n_part][n_angle] = min(self.max_orientation, max(-self.max_orientation, self.current_ang[n_trunk][n_part][n_angle]+incr))

    def _normalizedAction_to_action(self, action, type = "flex"):
        if type == "flex":
            return self.a_rotation*action + self.b_rotation
        else:
            return self.a_translation*action + self.b_translation

    def compute_action(self, actions, nb_step):
        list_incr = []
        for i, action in enumerate(actions[:-1]):
            value_goal = self._normalizedAction_to_action(action, type="flex")
            current_value = self.current_ang[self.n_trunk[i]][self.n_part[i]][self.n_angle[i]]
            incr = (value_goal-current_value)/nb_step
            if abs(incr)>self.max_incr_rot:
                if incr>=0:
                    incr = self.max_incr_rot
                else:
                    incr = -self.max_incr_rot
            list_incr.append(incr)

        value_goal = self._normalizedAction_to_action(actions[-1], type="trans")
        current_value = self.actuators[0][2].position.value[0][2]
        incr = (value_goal-current_value)/nb_step
        if abs(incr)>self.max_incr_trans:
            if incr>=0:
                incr = self.max_incr_trans
            else:
                incr = -self.max_incr_trans
        list_incr.append(incr)

        return list_incr

    def apply_action(self, list_incr):
        for i, incr in enumerate(list_incr[:-1]):
            self._modifyAngle(self.n_part[i], self.n_trunk[i], self.n_angle[i], incr)
            self._changeOrientation(self.n_part[i], self.n_trunk[i])
        self._changeTranslation(list_incr[-1])

    def onKeypressedEvent(self, event):
        key = event['key']

        if key == 'A':
            self.current_part = (self.current_part +1)%2
            print(" >> Actuator n°:", self.current_part)
        if key == 'B':
            self.current_trunk = (self.current_trunk +1)%2
            print(" >> Actuator n°:", self.current_trunk)

        if key == "L":
            self._changeTranslation(self.max_incr_trans)
        if key == "M":
            self._changeTranslation(-self.max_incr_trans)

        if ord(key) == 18:  #left
            self._modifyAngle(self.current_part, self.current_trunk, 0, self.max_incr_rot)
            self._changeOrientation(self.current_part, self.current_trunk)
        if ord(key) == 20:  #right
            self._modifyAngle(self.current_part, self.current_trunk, 0, -self.max_incr_rot)
            self._changeOrientation(self.current_part, self.current_trunk)

        if ord(key) == 19:  #left
            self._modifyAngle(self.current_part, self.current_trunk, 1, self.max_incr_rot)
            self._changeOrientation(self.current_part, self.current_trunk)
        if ord(key) == 21:  #right
            self._modifyAngle(self.current_part, self.current_trunk, 1, -self.max_incr_rot)
            self._changeOrientation(self.current_part, self.current_trunk)

class StartingPointController(Sofa.Core.Controller):

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.bar_orientation = kwargs["bar_orientation"]
        self.case = kwargs["case"]

        self.idx = 0
        self.max_idx = 50
        incr = None

        self.final_action = None

    def _noisy_action(self, action, std = 0.1):
        return [min(max(float(np.random.normal(a, std)),-1), 1) for a in action]

    def _case1(self, idx):
        assert -45 <= self.bar_orientation <= 20
        if idx in [0, 1, 2]:
            action = [0, 0, 0.7, 0, 0, 0, -0.7, 0, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 3<=idx<15:
            action = [0, 0, 0.7, 0, 0, 0, -0.7, 0, 0.7]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 15<=idx<20:
            if idx == 15:
                _action = [[0, 0, 0.3, 0, 0, 0, -0.7, 0, 0.8],
                           [0, 0, 0.7, 0, 0, 0, -0.3, 0, 0.8],
                           [0, 0, 0.3, 0, 0, 0, -0.3, 0, 0.8]]
                id_action = np.random.choice([0, 1, 2], p = [0.25, 0.25, 0.5])
                self.final_action = self._noisy_action(_action[id_action], std = 0.1)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx
    def _case2(self, idx):
        assert -20 <= self.bar_orientation <= 45
        if idx in [0, 1, 2]:
            action = [0, 0, -0.7, 0, 0, 0, 0.7, 0, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 3<=idx<15:
            action = [0, 0, -0.7, 0, 0, 0, 0.7, 0, 0.7]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 15<=idx<20:
            if idx == 15:
                _action = [[0, 0, -0.3, 0, 0, 0, 0.7, 0, 0.8],
                           [0, 0, -0.7, 0, 0, 0, 0.3, 0, 0.8],
                           [0, 0, -0.3, 0, 0, 0, 0.3, 0, 0.8]]
                id_action = np.random.choice([0, 1, 2], p = [0.25, 0.25, 0.5])
                self.final_action = self._noisy_action(_action[id_action], std = 0.1)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx
    def _case3(self, idx):
        assert 70 <= self.bar_orientation <= 110
        if idx in [0, 1, 2]:
            action = [0, 0, 0, -0.7, 0, 0, 0, 0.7, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 3<=idx<6:
            action = [0, 0, -0.7, -0.5, 0, 0, 0.7, 0.5, 0.8]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 6<=idx<9:
            if idx == 6:
                _action = [[0, 0, -0.5, 0.2, 0, 0, 0.5, -0.2, 0.8],
                          [0, 0, -0.5, 0, 0, 0, 0.5, -0.2, 0.8],
                          [0, 0, -0.5, 0.2, 0, 0, 0.5, 0, 0.8],
                          [0, 0, -0.5, -0.1, 0, 0, 0.5, 0.1, 0.8]]
                id_action = np.random.choice([0, 1, 2, 3], p = [0.4, 0.2, 0.2, 0.2])
                self.final_action = self._noisy_action(_action[np.random.randint(0,4)], std = 0.1)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx
    def _case4(self, idx):
        assert 70 <= self.bar_orientation <= 110
        if idx in [0, 1, 2]:
            action = [0, 0, 0, -0.7, 0, 0, 0, 0.7, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 3<=idx<6:
            action = [0, 0, 0.7, -0.5, 0, 0, -0.7, 0.5, 0.8]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 6<=idx<9:
            if idx == 6:
                _action = [[0, 0, 0.5, 0.2, 0, 0, -0.5, -0.2, 0.8],
                          [0, 0, 0.5, 0, 0, 0, -0.5, -0.2, 0.8],
                          [0, 0, 0.5, 0.2, 0, 0, -0.5, 0, 0.8],
                          [0, 0, 0.5, -0.1, 0, 0, -0.5, 0.1, 0.8]]
                id_action = np.random.choice([0, 1, 2, 3], p = [0.4, 0.2, 0.2, 0.2])
                self.final_action = self._noisy_action(_action[id_action], std = 0.1)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx
    def _case5(self, idx):
        assert 0 <= self.bar_orientation <= 45
        if idx in [0, 1, 2]:
            action = [0, 0, 0, -0.7, 0, 0, 0, 0.7, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 3<=idx<9:
            action = [0, 0, 0.7, 1, 0, 0, -0.7, -1, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 9<= idx < 14:
            action = [0, 0, 0.7, 1, 0, 0, -0.7, -1, 0.8]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 14<= idx < 16:
            action = [0.1, 0, 0.1, 0.6, -0.1, 0, -0.1, -0.6, 0.8]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 16<=idx<20:
            if idx == 16:
                _action = [ [0.1, 0, 0.1, 0.6, -0.1, 0, -0.1, -0.6, 0.8],
                           [0.1, 0, 0.1, 1, -0.1, 0, -0.1, -0.6, 0.8],
                           [0.1, 0, 0.1, 0.6, -0.1, 0, -0.1, -1, 0.8]]
                id_action = np.random.choice([0, 1, 2], p = [0.5, 0.25, 0.25])
                self.final_action = self._noisy_action(_action[np.random.randint(0,3)], std = 0.1)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx
    def _case6(self, idx):
        assert 0 <= self.bar_orientation <= 45
        if idx in [0, 1, 2]:
            action = [0, 0, 0, -0.7, 0, 0, 0, 0.7, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 3<=idx<9:
            action = [0, 0, -0.7, -1, 0, 0, 0.7, 1, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 9<= idx < 12:
            action = [0, 0, -0.7, -1, 0, 0, 0.7, 1, 0.8]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 12<= idx < 14:
            action = [0.1, 0, -0.7, 0.2, -0.1, 0, 0.7, -0.2, 0.8]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 14<= idx < 19:
            action = [0.1, -0.2, -0.4, 0.8, -0.1, 0.2, 0.4, -0.8, 0.9]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 19<=idx<20:
            if idx == 19:
                _action = [ [0.1, -0.2, -0.2, 0.8, -0.1, 0.2, 0.2, -0.8, 0.9],
                           [0.1, -0.2, -0.4, 0.8, -0.1, 0.2, 0.2, -0.8, 0.9],
                           [0.1, -0.2, -0.2, 0.8, -0.1, 0.2, 0.4, -0.8, 0.9]]
                id_action = np.random.choice([0, 1, 2], p = [0.5, 0.25, 0.25])
                self.final_action = self._noisy_action(_action[id_action], std = 0.1)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx
    def _case7(self, idx):
        assert 45 <= self.bar_orientation <= 90
        if idx in [0, 1, 2]:
            action = [0, 0, 0.7, 0, 0, 0, -0.7, 0, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 3<=idx<9:
            action = [0, 0, 0.7, 0.8, 0, 0, -0.7, -0.8, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 9<= idx < 14:
            action = [0, 0, 0.5, 0.8, 0, 0, -0.5, -0.8, 0]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 14<= idx < 18:
            action = [0.1, 0, 0.3, 0.8, -0.1, 0, -0.3, -0.8, 0.2]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 18<=idx<20:
            if idx == 18:
                _action = [ [0.1, 0, 0.1, 0.5, -0.1, 0, -0.1, -0.5, 0.8],
                           [0.1, 0, 0.1, 1, -0.1, 0, -0.1, -0.5, 0.8],
                           [0.1, 0, 0.1, 0.5, -0.1, 0, -0.1, -1, 0.8]]
                id_action = np.random.choice([0, 1, 2], p = [0.5, 0.25, 0.25])
                self.final_action = self._noisy_action(_action[id_action], std = 0.1)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx
    def _case8(self, idx):
        assert 130 <= self.bar_orientation <= 140
        if idx in [0, 1, 2]:
            action = [0, 0, -0.7, 0, 0, 0, 0.7, 0, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 3<=idx<9:
            action = [0, 0, -0.7, 0.7, 0, 0, 0.7, -0.7, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 9<= idx < 14:
            action = [0, 0, -0.7, 0.7, 0, 0, 0.7, -0.7, 0.8]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 14<= idx < 18:
            action = [0.1, 0, -0.3, 0.6, -0.1, 0, 0.3, -0.6, 0.8]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 18<=idx<20:
            if idx == 18:
                _action = [[0.1, 0, -0.1, 0.5, -0.1, 0, 0.1, -0.5, 0.8],
                           [0.1, 0, -0.1, 1, -0.1, 0, 0.1, -0.5, 0.8],
                           [0.1, 0, -0.1, 0.5, -0.1, 0, 0.1, -1, 0.8]]
                id_action = np.random.choice([0, 1, 2], p = [0.5, 0.25, 0.25])
                self.final_action = self._noisy_action(_action[id_action], std = 0.1)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx
    def _case9(self, idx):
        assert 0 <= self.bar_orientation <= 180
        if idx<=10:
            if idx == 0:
                self.final_action = [0, 0, 0, 0, 0, 0, 0, 0, -1]
            self.final_action = self._noisy_action(self.final_action, std = 0.3)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx
    def _case10(self, idx):
        assert -20 <= self.bar_orientation <= 20
        if idx in [0, 1, 2]:
            action = [0, 0, 0.7, 0, 0, 0, 0.7, 0, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 3<=idx<15:
            action = [0, 0, 0.7, 0, 0, 0, 0.7, 0, 0.7]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 15<=idx<20:
            if idx == 15:
                _action = [[0, 0, 0.3, 0, 0, 0, 0.7, 0, 0.8],
                           [0, 0, 0.7, 0, 0, 0, 0.3, 0, 0.8],
                           [0, 0, 0.3, 0, 0, 0, 0.3, 0, 0.8]]

                self.final_action = self._noisy_action(_action[np.random.randint(0,3)], std = 0.1)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx
    def _case11(self, idx):
        assert -20 <= self.bar_orientation <= 20
        if idx in [0, 1, 2]:
            action = [0, 0, -0.7, 0, 0, 0, -0.7, 0, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 3<=idx<15:
            action = [0, 0, -0.7, 0, 0, 0, -0.7, 0, 0.7]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 15<=idx<20:
            if idx == 15:
                _action = [[0, 0, -0.3, 0, 0, 0, -0.7, 0, 0.8],
                           [0, 0, -0.7, 0, 0, 0, -0.3, 0, 0.8],
                           [0, 0, -0.3, 0, 0, 0, -0.3, 0, 0.8]]
                self.final_action = self._noisy_action(_action[np.random.randint(0,3)], std = 0.1)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx
    def _case12(self, idx):
        assert 85 <= self.bar_orientation <= 95
        if idx in [0, 1, 2]:
            action = [0, 0, 0.7, 0.7, 0, 0, -0.7, 0, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 3<=idx<15:
            action = [0, 0, 0.3, 1, 0, 0, -0.7, 0, 0.3]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 15<=idx<18:
            action = [0, 0, 0.1, 1, 0, 0, -0.7, 0, 0.7]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 18<=idx<20:
            if idx == 18:
                _action = [[0, 0, 0, 0.1, 0, 0, -0.7, 0, 0.8],
                           [0, 0, 0, 1, 0, 0, -0.1, 0, 0.8],
                           [0, 0, 0, 0.1, 0, 0, -0.1, 0, 0.8]]
                self.final_action = self._noisy_action(_action[np.random.randint(0,3)], std = 0.1)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx
    def _case13(self, idx):
        assert 85 <= self.bar_orientation <= 95
        if idx in [0, 1, 2]:
            action = [0, 0, -0.7, -0.1, 0, 0, 0.7, -0.7, -1]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 3<=idx<15:
            action = [0, 0, -0.7, -0.1, 0, 0, 0.3, -1, 0.3]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 15<=idx<18:
            action = [0, 0, -0.7, 0, 0, 0,  0.1, -1, 0.7]
            action = self._noisy_action(action, std = 0.1)
            incr = self.root.applyAction.compute_action(action, 1)
            self.root.applyAction.apply_action(incr)
        elif 18<=idx<20:
            if idx == 18:
                _action = [[0, 0, -0.7, 0, 0, 0, 0, -0.1,0.8],
                           [ 0, 0, -0.1, 0, 0, 0, 0, -1, 0.8],
                           [0, 0, -0.1, 0, 0, 0, 0, -0.1, 0.8]]
                self.final_action = self._noisy_action(_action[np.random.randint(0,3)], std = 0.1)
            incr = self.root.applyAction.compute_action(self.final_action, 1)
            self.root.applyAction.apply_action(incr)
        else:
            self.idx = self.max_idx

    def onAnimateBeginEvent(self, eventType):
        if self.idx == 0:
            self.root.Reward.update()
        if self.idx < self.max_idx:
            if self.case == 1:
                self._case1(self.idx)
            elif self.case == 2:
                self._case2(self.idx)
            elif self.case == 3:
                self._case3(self.idx)
            elif self.case == 4:
                self._case4(self.idx)
            elif self.case == 5:
                self._case5(self.idx)
            elif self.case == 6:
                self._case6(self.idx)
            elif self.case == 7:
                self._case7(self.idx)
            elif self.case == 8:
                self._case8(self.idx)
            elif self.case == 9:
                self._case9(self.idx)
            elif self.case == 10:
                self._case10(self.idx)
            elif self.case == 11:
                self._case11(self.idx)
            elif self.case == 12:
                self._case12(self.idx)
            elif self.case == 13:
                self._case13(self.idx)
            self.idx+=1
        # _,_, ms = self.root.Reward.getReward()
        # print("[INFO] >> MS:", ms)
