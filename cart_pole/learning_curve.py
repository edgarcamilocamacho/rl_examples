# //======================================================================//
# //  This software is free: you can redistribute it and/or modify        //
# //  it under the terms of the GNU General Public License Version 3,     //
# //  as published by the Free Software Foundation.                       //
# //  This software is distributed in the hope that it will be useful,    //
# //  but WITHOUT ANY WARRANTY; without even the implied warranty of      //
# //  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE..  See the      //
# //  GNU General Public License for more details.                        //
# //  You should have received a copy of the GNU General Public License   //
# //  Version 3 in the file COPYING that came with this distribution.     //
# //  If not, see <http://www.gnu.org/licenses/>                          //
# //======================================================================//
# //                                                                      //
# //      Copyright (c) 2020 - Edgar Camilo Camacho Poveda                //   
# //      camilo.im93@gmail.com                                           //
# //                                                                      //
# //======================================================================//

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg  
import pyqtgraph.exporters

class LearningCurve:

    def __init__(   self, 
                    plots,
                    title='Learning Curve', 
                    episode_range=100, 
                    label_left = 'Utility',
                    min_y_left = -1.0, 
                    max_y_left = 1.0,
                    label_right = 'Epsilon',
                    min_y_right = 0.0, 
                    max_y_right = 1.0,
                    line_width = 2.0
                    ):
                    
        self.episode_range = episode_range
        self.label_left = label_left
        self.min_y_left = min_y_left
        self.max_y_left = max_y_left
        self.label_right = label_right
        self.min_y_right = min_y_right
        self.max_y_right = max_y_right
        self.line_width = line_width

        pg.mkQApp()
        self.pw = pg.PlotWidget()
        self.pw.show()
        self.pw.setWindowTitle(title)
        self.p1 = self.pw.plotItem
        self.p1.setLabels(left=self.label_left)
        self.p1.setXRange(0, self.episode_range)
        self.p1.setYRange(self.min_y_left, self.max_y_left)
        self.p1.showGrid(x = True, y = True, alpha = 0.3) 
        self.p1.setLabel('bottom', 'Episodes')


        self.p2 = pg.ViewBox()
        self.p1.showAxis('right')
        self.p1.scene().addItem(self.p2)
        self.p1.getAxis('right').linkToView(self.p2)
        self.p2.setXLink(self.p1)
        self.p1.getAxis('right').setLabel(self.label_right, color='#0000ff')

        self.p2.setYRange(self.min_y_right, self.max_y_right)

        self.curves = {}
        self.X = {}
        self.Y = {}

        for plot in plots:
            plot_id = plot[0]
            if plot[1] == 'left':
                self.curves[plot_id] = self.p1.plot( pen=pg.mkPen(plot[2], width=self.line_width) )
            else:
                self.curves[plot_id] = pg.PlotCurveItem( pen=pg.mkPen(plot[2], width=self.line_width) )
                self.p2.addItem(self.curves[plot_id])
            self.X[plot_id] = np.array([])
            self.Y[plot_id] = np.array([])
            self.curves[plot_id].setData(self.X[plot_id], self.Y[plot_id])

        QtGui.QApplication.processEvents()

    def add_sample(self, plot_ids, episode, ys):
        for i, plot_id in enumerate(plot_ids):
            self.X[plot_id] = np.append(self.X[plot_id], episode)
            self.Y[plot_id] = np.append(self.Y[plot_id], ys[i])
            self.curves[plot_id].setData(self.X[plot_id], self.Y[plot_id])
        if episode>self.episode_range:
            self.p1.setXRange(0, episode)
        self.p2.setGeometry(self.p1.vb.sceneBoundingRect())
        QtGui.QApplication.processEvents()
    
    def save_plot(self, name):
        exporter = pg.exporters.ImageExporter(self.p1)
        exporter.parameters()['width'] = 1024
        if not name.endswith('.png'):
            name += '.png'
        exporter.export(name)