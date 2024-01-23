# -*- coding: utf-8 -*-
"""Create a viewer to observe Knowledge graph and explain the
decision of the agent.
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Jun 17 2021"


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

class GraphVisualisation():
    def __init__(self, ax):
        self.G = nx.DiGraph()
        self.pos = nx.planar_layout(self.G)
        self.active_color, self.non_active_color, self.target_color = "red", "blue", "green"
        self.colors_node = dict()
        self.colors_edge = dict()
        self.proba = dict()
        self.idx = 0

        self.ax = ax

    def add_node(self, proba):
        idx = self.idx
        self.idx+=1
        self.G.add_node(idx)
        self.colors_node.update({idx: self.non_active_color})
        self.proba.update({idx: proba})
        self.pos = nx.planar_layout(self.G)

        return idx

    def add_edge(self, from_node, to_node):
        self.G.add_edge(from_node, to_node)
        self.colors_edge.update({str(from_node)+"-"+str(to_node): "black"})

    def remove_node(self, idx):
        succ = list(self.G.successors(idx))
        pred = list(self.G.predecessors(idx))
        for s in succ:
            self.G.remove_edge(idx, s)
            self.colors_edge.pop(str(idx)+"-"+str(s))
        for p in pred:
            if p!= idx:
                self.G.remove_edge(p, idx)
                self.colors_edge.pop(str(p)+"-"+str(idx))
        self.G.remove_node(idx)
        self.proba.pop(idx)
        self.colors_node.pop(idx)

    def change_active(self, idx_old, idx_new):
        self.colors_node[idx_old] = self.non_active_color
        self.colors_node[idx_new] = self.active_color

    def target_node(self,  from_node, to_node, reverse = False):
        if reverse:
            self.colors_node[to_node] = self.non_active_color
            self.colors_edge.update({str(from_node)+"-"+str(to_node): "black"})
        else:
            self.colors_node[to_node] = self.target_color
            self.colors_edge.update({str(from_node)+"-"+str(to_node): self.target_color})

    def create_colors(self):
        nodes = list(self.G.nodes)
        n_color = []
        for n in nodes:
            n_color.append(self.colors_node[n])

        edges = list(self.G.edges)
        e_color = []
        for (p,s) in edges:
            e_color.append(self.colors_edge[str(p)+"-"+str(s)])

        return n_color, e_color


    def change_proba(self, idx, new_proba):
        self.proba[idx] = new_proba

    def plot(self, action = None):
        self.ax.clear()
        self.ax.set_title('Knowledge graph', fontweight='bold')
        n_color, e_color = self.create_colors()
        nx.draw(self.G, self.pos, with_labels = True,
                    ax = self.ax,
                    connectionstyle='arc3, rad = 0.1',
                     font_weight='bold',
                     labels = self.proba,
                     node_color= n_color,
                     edge_color= e_color)

        if action is not None:
            self.addAction(str(action))

    def addAction(self, action, pos = (-1.2, -1)):
        self.ax.text(pos[0], pos[1],"Selected action: " + action, bbox=dict(facecolor='green', alpha=0.5))

class DisplayImages():
    def __init__(self, ax, title = ""):
        self.ax = ax
        self.title = title

    def display(self, image = np.ones((200, 300)), text= None):
        self.ax.clear()
        self.ax.set_title(self.title, fontweight='bold')
        self.ax.imshow(image)


def create_fig(size = (14, 10)):
    plt.ion()

    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, :])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.show()

    return ax0, ax1, ax2, ax3


class Viewer():
    def __init__(self):
        # ax0, ax1, ax2, ax3 = create_fig(size = (14, 10))
        # self.D1 = DisplayImages(ax1, 'Current state x_t')
        # self.D1.display()
        # self.D2 = DisplayImages(ax2, 'Examples of states in the considered class')
        # self.D2.display()
        # self.D3 = DisplayImages(ax3, 'Examples of states in the next class')
        # self.D3.display()
        #
        # self.G = GraphVisualisation(ax0)
        self.G = GraphVisualisation(None)
    def updateColorG(self, active_node, old_active_node, target_node = None):
        self.G.change_active(old_active_node, active_node)
        if target_node is not None:
            self.G.target_node(active_node, target_node)

    def updateNodeG(self, p, remove = (False, None)):
        if remove[0]:
            self.G.remove_node(remove[1])
            return -1
        return self.G.add_node(p)

    def updateEdgeG(self, edges = []):
        for (p,s) in edges:
            self.G.add_edge(p, s)

    def updateProbaG(self, idx = [], p = []):
        l = len(idx)
        assert l==len(p)
        for i in range(l):
            self.G.change_proba(idx[i], p[i])

    def plot(self, waiting_time = 0.1, action = None, image_current= np.ones((200, 300)), image_class= np.ones((200, 300)), image_next= np.ones((200, 300))):
        self.D1.display(image_current)
        self.D2.display(image_class)
        self.D3.display(image_next)
        self.G.plot(action = action)

        plt.draw()
        plt.pause(waiting_time)



if __name__ == '__main__':
    Viewer = Viewer()

    idx_gauche = Viewer.updateNodeG(0)
    idx_droit = Viewer.updateNodeG(0)
    idx_centre = Viewer.updateNodeG(0)

    edges = [(idx_gauche, idx_droit), (idx_droit, idx_gauche), (idx_centre, idx_droit),
            (idx_centre, idx_gauche), (idx_gauche, idx_centre), (idx_droit, idx_centre),
            (idx_droit, idx_droit), (idx_centre, idx_centre), (idx_gauche, idx_gauche)]

    Viewer.updateEdgeG(edges)

    Viewer.plot(waiting_time = 5)

    Viewer.updateColorG(idx_centre, idx_gauche, idx_gauche)
    Viewer.updateProbaG([idx_centre], [1])

    Viewer.plot(waiting_time = 5, action = [1, 2, 3, 4, 5, 6, 7, 8])
