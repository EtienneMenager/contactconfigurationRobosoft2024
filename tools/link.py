# -*- coding: utf-8 -*-
"""Create action to define between which nodes
we have an edge.
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 23 2021"

from operator import itemgetter

class Link():
    def __init__(self, from_node, to_node, max_len = 30):
        self.from_node = from_node
        self.to_node = to_node
        self.max_len = max_len
        self.Buffer = []

    def updateBuffer(self, new_actions = [], weight = None):
        self.Buffer = self.Buffer + new_actions
        if weight is not None and len(self.Buffer)== len(weight):
            inter = list(zip(self.Buffer, weight))
            inter.sort(key=itemgetter(1))
            self.Buffer, _ = zip(*inter)
        self.Buffer = self.Buffer[:self.max_len]
