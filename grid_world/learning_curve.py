import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

class LearningCurve:

    def __init__(self, title='Learning Curve', eps_range=100, min_y=-1.0, max_y=1.0):
        self.eps_range = eps_range
        self.min_y = min_y
        self.max_y = max_y
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title=title)
        self.p = self.win.addPlot(title=title)
        self.p.setXRange(0, self.eps_range)
        self.p.setYRange(self.min_y, self.max_y)
        self.p.showGrid(x = True, y = True, alpha = 0.3) 
        self.p.setLabel('left', 'Utility')
        self.p.setLabel('bottom', 'Episodes')
        self.curve = self.p.plot(pen='r')
        self.X = np.array([])
        self.Y = np.array([])
        self.curve.setData(self.X,self.Y)
        QtGui.QApplication.processEvents()

    def add_sample(self, episode, y):
        self.X = np.append(self.X, episode)
        self.Y = np.append(self.Y, y)
        self.curve.setData(self.X,self.Y)
        if episode>self.eps_range:
            self.p.setXRange(0, episode)
        QtGui.QApplication.processEvents()