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
from pyquaternion import Quaternion
from operator import itemgetter
import itertools

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

SofaRuntime.importPlugin("SofaComponentAll")

class rewardShaper(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.verbose = kwargs["verbose"]
        self.tot_ang = kwargs["init_rot"]


    def getReward(self):
        current_ref = self._getBarVec()
        vector_value = np.max([-1., np.min([1., np.dot(self.pred_ref, current_ref)/(np.linalg.norm(self.pred_ref)*np.linalg.norm(current_ref))])])
        angle = np.degrees(np.arccos(vector_value))
        signe = np.sign(np.cross( current_ref/np.linalg.norm(current_ref), self.pred_ref/np.linalg.norm(self.pred_ref))[-1])
        self.pred_ref = current_ref
        _ang = float(angle*signe)
        self.tot_ang += _ang
        _tot_ang = round(self.tot_ang, 1)
        current_ang_dist = abs(self.ang_goal-self.tot_ang)

        reward_ang = -current_ang_dist/abs(self.ang_goal)
        #reward_ang = (self.pred_ang_dist - current_ang_dist)/abs(self.ang_goal)
        #self.pred_ang_dist = current_ang_dist

        barPos = self.root.sceneModerator.bar.effectors.EffectorsMO.position.value[:, :3]
        A, B = barPos[0], barPos[-1]
        sensor_0 = self.root.sceneModerator.trunks[0].sensors.SensorMO.position.value[:]
        sensor_1 = self.root.sceneModerator.trunks[1].sensors.SensorMO.position.value[:]
        dist_0 = min(self._compute_distance_trunk_bar(A, B, sensor_0))
        dist_1 = min(self._compute_distance_trunk_bar(A, B, sensor_1))
        control_dist = 30
        reward_dist = min(0, (control_dist - dist_0)/control_dist) + min(0, (control_dist - dist_1)/control_dist)

        trunks = self.root.sceneModerator.trunks
        control_crois = 20
        sensor_0 = trunks[0].sensors.SensorMO.position.value[:]
        sensor_1 = trunks[1].sensors.SensorMO.position.value[:]
        x_pos_tips_0 = sensor_0.mean(axis = 0)[0]
        x_pos_tips_1 = sensor_1.mean(axis = 0)[0]
        reward_tips = min((x_pos_tips_0 - x_pos_tips_1)/control_crois, 0)

        factor = [3, 1/4, 1/8]
        reward = factor[0]*reward_ang + factor[1]*reward_dist #+ factor[2]*reward_tips
        #reward = reward_ang

        dist = abs(self.ang_goal-_tot_ang)
        if self.verbose:
            print("\n[INFO] >> Reward:", reward)
            print("[INFO] >> Angle:", _ang)
            print("[INFO] >> Total angle:", _tot_ang)
            print("[INFO] >> Target:", self.ang_goal)
            print("[INFO] >> Reward Angle:", reward_ang)
            print("[INFO] >> Dist:", ((control_dist - dist_0)/control_dist), " - ", ((control_dist - dist_1)/control_dist))
            print("[INFO] >> Dist_0 reward:", min(0, (control_dist - dist_0)/control_dist))
            print("[INFO] >> Dist_1 reward:", min(0, (control_dist - dist_1)/control_dist))
            print("[INFO] >> Tips:", x_pos_tips_0, " - ", x_pos_tips_1)
            print("[INFO] >> Reward Tips:", reward_tips)


        if np.isinf(dist) or np.isnan(dist) or np.isinf(reward) or np.isnan(reward):
            print("[ERROR] >> In the reward:")
            print(">>  dist:", dist)
            print(">>  reward:", reward)
            print(">>  reward ang:", reward_ang)
            print(">>  reward dist:", reward_dist)
            exit(1)

        return reward, dist, None #self._getMS()

    def update(self):
        self.pred_ref = self._getBarVec()
        self.ang_goal = self.root.GoalSetter.goalPos
        self.pred_ang_dist = abs(self.ang_goal - self.tot_ang)

    def _compute_distance_trunk_bar(self, A, B, P):
        AB = (B - A)
        norm_AB = np.linalg.norm(AB)
        AB = AB/norm_AB
        AP = P - A

        dot = np.dot(AP, AB)
        #If we are not beetween A and B
        dot = np.maximum([0 for _ in range(dot.shape[0])], dot)
        dot = np.minimum([norm_AB for _ in range(dot.shape[0])], dot)
        dot = dot.reshape(1, -1)

        value = np.repeat(dot, 3, axis = 0)
        C = A + value.T*AB
        norm = np.linalg.norm(P-C, axis = 1)

        return norm

    def _getBarVec(self):
        bar_point_A= self.root.sceneModerator.bar.effectors.EffectorsMO.position.value[0, :3]
        bar_point_B = self.root.sceneModerator.bar.effectors.EffectorsMO.position.value[2, :3]
        v_bar = (bar_point_B-bar_point_A)/np.linalg.norm(bar_point_B-bar_point_A)
        return np.cross(v_bar, np.array([0, 0, 1]))

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

    def _compute_ms(self, distances_0, distances_1, print_info = False, tresh_xy = [6, 16], tresh_z = 11, tresh_along_bar = 0):
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

    def _getMS(self):
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
        ms = self._compute_ms([bar_side_0, dist_xy_0, dist_z_0, along_bar_0],
                          [bar_side_1, dist_xy_1, dist_z_1, along_bar_1], print_info = self.verbose)

        return ms

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

        self.bar=  kwargs["bar"]
        self.trunks=  kwargs["trunks"]
        self.actuators=  kwargs["actuators"]

    def getPos(self):
        pos_trunk_0 = self.trunks[0].getPos()
        pos_trunk_1 = self.trunks[1].getPos()

        pos_bar = self.bar.getPos()
        return [pos_trunk_0, pos_trunk_1, pos_bar]

    def setPos(self, pos):
        [pos_trunk_0, pos_trunk_1, pos_bar] = pos

        self.trunks[0].setPos(pos_trunk_0)
        self.trunks[1].setPos(pos_trunk_1)
        self.bar.setPos(pos_bar)

################################################################################

def compute_dist(A, B, P, treshold):
    AB = (B - A)
    norm_AB = np.linalg.norm(AB)
    AB = AB/norm_AB
    AP = P - A

    dot = np.dot(AP, AB)
    #If we are not beetween A and B
    dot = np.maximum([0 for _ in range(dot.shape[0])], dot)
    dot = np.minimum([norm_AB for _ in range(dot.shape[0])], dot)
    dot = dot.reshape(1, -1)

    value = np.repeat(dot, 3, axis = 0)
    C = A + value.T*AB
    norm = np.linalg.norm(P-C, axis = 1)
    tresh_norm = np.maximum([0 for _ in range(norm.shape[0])], (treshold-norm)/treshold)
    return tresh_norm.tolist()

def getState(rootNode):
    """Compute the state of the environment/agent.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        State: list of float
            The state of the environment/agent.
    """
    bar = rootNode.sceneModerator.bar
    trunks = rootNode.sceneModerator.trunks

    #Position of the effectors for the 2 trunks: 2*3*nb_point  avec nb_point = 3
    effectorPos_0 = trunks[0].effector.EffectorMO.position.value[:, :3].reshape((-1)).tolist()
    effectorPos_1 = trunks[1].effector.EffectorMO.position.value[:, :3].reshape((-1)).tolist()

    #Position of the point of the beam: 3*3 + orientation if we want

    barPos = bar.effectors.EffectorsMO.position.value[:, :3]
    mid_point =  barPos[1]

    # if barPos[0][0] <= barPos[-1][0]:
    #     left_point, right_point =  barPos[0], barPos[-1]
    # else:
    #     left_point, right_point =  barPos[-1], barPos[0]
    # # left_point, right_point = barPos[0], barPos[-1]
    # barPos = left_point.tolist() + mid_point.tolist() + right_point.tolist()

    barPos = mid_point.tolist()

    barPos = [p / 150 if i % 3 == 2 else p / 100 for i, p in enumerate(barPos)]
    barPos+= [rootNode.Reward.tot_ang]
    # A, B = barPos[0], barPos[-1]
    # barPos = barPos.reshape((-1)).tolist()


    # #Distance between sensor and beam for each trunks: 2*4
    # sensor_0 = trunks[0].sensors.SensorMO.position.value[:]
    # sensor_1 = trunks[1].sensors.SensorMO.position.value[:]
    #
    # dist_0 = compute_dist(A, B, sensor_0, 30)
    # dist_1 = compute_dist(A, B, sensor_1, 30)

    # #Pressure in cavity: 2*1
    # v_1 = trunks[0].cavity.SurfacePressureConstraint.volumeGrowth.value
    # v_2 = trunks[1].cavity.SurfacePressureConstraint.volumeGrowth.value
    # volumeGrowth = [v_1, v_2]

    effectorPos_0 = [p for p in effectorPos_0]
    effectorPos_1 = [p for p in effectorPos_1]
    # volumeGrowth = [v for v in volumeGrowth]

    effectorPos_0 = [p/150 if i%3==2 else p/100 for i, p in enumerate(effectorPos_0)]
    effectorPos_1 = [p/150 if i%3==2 else p/100 for i, p in enumerate(effectorPos_1)]

    # volumeGrowth = [v/300 for v in volumeGrowth]
    goal_pos = [rootNode.GoalSetter.goalPos]

    # state = effectorPos_0 + effectorPos_1 + barPos + dist_0 + dist_1 + volumeGrowth + goal_pos
    state = effectorPos_0 + effectorPos_1 + barPos  + goal_pos

    if np.any(np.isinf(state)) or np.any(np.isnan(state)):
        print("[ERROR] >> In the observation:")
        print(">>  effectorPos_0:", effectorPos_0)
        print(">>  effectorPos_1:", effectorPos_1)
        print(">>  barPos:", barPos)
        # print(">>  dist_0:", dist_0)
        # print(">>  dist_1:", dist_1)
        # print(">>  volumeGrowth:", volumeGrowth)
        print(">>  goal_pos:", goal_pos)
        exit(1)

    return state


def getReward(rootNode):
    r, diff_ang, _ =  rootNode.Reward.getReward()
    done = diff_ang < 5
    return done, r


def getPos(root):
    position = root.sceneModerator.getPos()
    return position


def getInfos(root):
    return root.Reward._getMS(), root.Reward.tot_ang

def setPos(root, pos):
    root.sceneModerator.setPos(pos)

################################################################################

class applyAction(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.actuators = kwargs['actuators']
        self.max_incr_rot = kwargs["max_incr_rot"]
        self.max_orientation = kwargs["max_orientation"]
        self.max_incr_trans = kwargs["max_incr_trans"]
        self.max_translation= kwargs["max_translation"]
        self.init_action = kwargs["init_action"]

        self.a_rotation, self.b_rotation =  self.max_orientation, 0
        self.a_translation, self.b_translation = -self.max_translation/2,  -self.max_translation/2

        self.n_trunk = [0, 0, 0, 0, 1, 1, 1, 1]
        self.n_part =  [0, 0, 1, 1, 0, 0, 1, 1]
        self.n_angle = [0, 1, 0, 1, 0, 1, 0, 1]

        self.current_part = 0
        self.current_trunk = 0

        self.current_ang = np.zeros(shape = (2, 2, 3))
        for i in range(2):
            self.current_ang[i][0][2] = 180
            self.current_ang[i][1][2] = 180

        self.current_id = 0
        self.max_id = len(self.init_action)

    def onAnimateBeginEvent(self, event):
        if self.current_id < self.max_id:
            action = self.init_action[self.current_id]
            incr = self.compute_action(action, 1)
            self.apply_action(incr)
            self.current_id+=1

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



def action_to_command(actions, root, nb_step):
    incr = root.applyAction.compute_action(actions, nb_step)
    return incr

def startCmd(root, actions, duration):
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
