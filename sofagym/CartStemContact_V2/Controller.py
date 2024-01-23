# -*- coding: utf-8 -*-
"""Controller for the Abstraction of Jimmy.


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

class ControllerCartStem(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root =  kwargs["root"]
        if "cartstem" in kwargs:
            print(">>  Init cartstem...")
            self.cartstem = kwargs["cartstem"]
        else:
            print(">>  No cartstem ...")
            self.cartstem = None

        self.incr = 0.1
        print(">>  Init done.")
        self.init_reward = False

    def onAnimateBeginEvent(self, event):
        contacts = self.root.sceneModerator.contacts
        factor = self.cartstem.max_move

        posCart = self.cartstem.cartStem.MechanicalObject.position.value.tolist()[0][0]/factor
        posTip =  self.cartstem.cartStem.MechanicalObject.position.value.tolist()[-1][0]/factor
        posContacts = [p/factor for p in contacts.getPos()]
        goal = self.root.GoalSetter.goalPos[0]/factor
        state = [posCart, posTip] + posContacts + [goal]

        print("\n>> posCart: ", posCart)
        print(">> posTip: ", posTip)
        print(">> posContacts: ", posContacts)
        print(">> goal: ", [goal])
        print(">> MAX MOVE:", self.cartstem.max_move)
        print(state)

        if self.init_reward:
            print("\n>> Reward / dist: ", self.root.Reward.getReward())
            print(">> Init dist: ", self.root.Reward.init_goal_dist)
        else:
            self.root.Reward.update()
            self.init_reward = True

    def _move(self, incr):
        controlMO = self.root.rest_shape.goalMO
        with controlMO.position.writeable() as pos:
            pos[0,0]+=incr

    def onKeypressedEvent(self, event):
        key = event['key']
        if ord(key) == 18:  #left
            self._move(-self.incr)
        if ord(key) == 20:  #right
            self._move(self.incr)

        if key == 'A':
            pos = self.root.sceneModerator.getPos()
            with open("./pos.txt", 'w') as fp:
                json.dump([pos[0]], fp)

            with open("./cube_pos.txt", 'w') as fp:
                json.dump([pos[1]], fp)

            with open("./collis_pos.txt", 'w') as fp:
                json.dump([pos[2]], fp)
