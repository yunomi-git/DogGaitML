# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:51:38 2021

@author: Evan Yu
"""
from abc import ABC, abstractmethod
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from PyQt5.QtWidgets import QSlider

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget
    
class Slider(QWidget):
    def __init__(self, minimum, maximum, parent=None):
        super(Slider, self).__init__(parent=parent)
        
        self.verticalLayout = QVBoxLayout(self)
        self.horizontalLayout = QHBoxLayout()
        self.label = QLabel(self)
        self.horizontalLayout.addWidget(self.label)
        
        # spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        # self.horizontalLayout.addItem(spacerItem)
        
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setTickInterval(1.0)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue(self.slider.value())

    def setLabelValue(self, value):
        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
        self.maximum - self.minimum)
        self.x = int(self.x)
        self.label.setText("{0:.4g}".format(self.x))


class GeneticVisualizer():
    def __init__(self, costEvaluator, dataHistory, convergenceHistory):
       self.setDefaultOptions()
       self.costEvaluator = costEvaluator
       self.historyDisplayIndex = 0
       self.setDataHistory(dataHistory)
       
       self.convergenceHistory = self.fixConvergenceHistory(convergenceHistory)
       
       # self.createEverything()
       
    def setDefaultOptions(self):
        self.setPlotRange(xMax = 15, yMax = 15)
        self.setResolution(200)
        
    def fixConvergenceHistory(self, convergenceHistory):
        convergenceHistory = np.array(convergenceHistory)
        minima = self.costEvaluator.getKnownGlobalMinima()
        minVal = minima[0][2]
        convergenceHistory -= minVal
        convergenceHistory = np.sign(convergenceHistory) * np.log(np.abs(convergenceHistory)/10. + 1)
        return convergenceHistory.tolist()
       
    def setPlotRange(self, xMax, yMax):
        self.xMax = xMax
        self.yMax = yMax
        self.avgGridSize = (xMax + yMax) / 2.0
        
    def setResolution(self, resolution):
        self.resolution = resolution
        
    # dataHistory is a list of np arrays with shape (numElements, 3)
    def setDataHistory(self, dataHistory):
        self.numHistory = len(dataHistory)
        self.dataHistory = dataHistory
        
    def visualize(self):
        self.wMain = pg.LayoutWidget()
        self.wMain.setWindowTitle('2d cost function')

        self.w3d = gl.GLViewWidget()
        self.w3d.setCameraPosition(distance=3 * self.avgGridSize)
        
        self.createCostSurface()
        self.createDataPlot()
        self.createGrid()
        self.createButtons()
        self.createOptimaPlot()
        
        self.createConvergencePlot()
        
        self.convPl.sizeHint = lambda: pg.QtCore.QSize(50, 100)
        self.w3d.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        self.w3d.setSizePolicy(self.convPl.sizePolicy())
        
        self.createSlider()
        self.createCostLabel()
        
        self.wMain.addWidget(self.prevBtn, row=0, col=0)
        self.wMain.addWidget(self.nextBtn, row=0, col=1)
        self.wMain.addWidget(self.convPl, row = 3, col = 0, colspan = 2)
        self.wMain.addWidget(self.w3d, row = 2, col = 0, colspan = 2)
        self.wMain.addWidget(self.slider, row = 1, col = 0, colspan = 2)
        self.wMain.addWidget(self.cost, row = 4, col = 0)
        self.wMain.addWidget(self.ideal, row = 4, col = 1)
        
        self.wMain.show()
        self.wMain.resize(800,800)
        
    def createCostLabel(self):
        self.cost = QLabel()
        self.ideal = QLabel()
        minima = self.costEvaluator.getKnownGlobalMinima()
        self.ideal.setText("Ideal: " + "{0:.5g}".format(minima[0][2]))
        
    def createCostSurface(self):
        x, y, z = self.getMeshCost()
    
        colors = np.empty((self.resolution, self.resolution, 4), dtype=np.float32)
        dataMax = z.max()
        dataMin = z.min()
        dataScale = np.abs(dataMax - dataMin)
        heightLayer = (z - dataMin) / dataScale
        uniformLayer = np.ones([self.resolution, self.resolution])
        alpha = np.ones([self.resolution, self.resolution]) * 0.5
    
        colors[:,:, 0] = heightLayer
        colors[:,:, 1] = uniformLayer * 0.4
        colors[:,:, 2] = uniformLayer * 0.4
        colors[:,:, 3] = alpha
        
        p = gl.GLSurfacePlotItem(x=x, y=y, z=z, colors=colors, shader = 'shaded')
        self.w3d.addItem(p)
        
    def getMeshCost(self):
        x = np.linspace(-self.xMax,self.xMax,self.resolution);
        y = np.linspace(-self.yMax,self.yMax,self.resolution);
        data = self.costEvaluator.getCostMesh(x, y)
        return x, y, data
        
    def createButtons(self):  
        self.nextBtn = QtGui.QPushButton('+1')
        self.prevBtn = QtGui.QPushButton('-1')
        
        self.nextBtn.clicked.connect(self.nextButton)
        self.prevBtn.clicked.connect(self.prevButton)
        

    
    def nextButton(self):
        self.incrementHistory(1)
    def prevButton(self):
        self.incrementHistory(-1)
        
    def createSlider(self):
        self.slider = Slider(0, self.numHistory - 1)
        self.slider.slider.valueChanged.connect(self.sliderUpdate)
        
    def sliderUpdate(self):
        idx = self.slider.x
        
        self.setHistoryIndex(idx)
        
    def incrementHistory(self, i):        
        historyDisplayIndex = self.historyDisplayIndex + i
        if historyDisplayIndex > (self.numHistory - 1):
            historyDisplayIndex = self.numHistory - 1
        if historyDisplayIndex < 0:
            historyDisplayIndex = 0
        self.setHistoryIndex(historyDisplayIndex)
        
    def setHistoryIndex(self, idx):
        self.historyDisplayIndex = idx
        prevhistoryDisplayIndex = self.historyDisplayIndex - 1
        if prevhistoryDisplayIndex < 0:
            prevhistoryDisplayIndex = 0
        prevData = self.dataHistory[prevhistoryDisplayIndex]
        self.prevDataPlot.setData(pos=prevData, color=(0,1,0,0.5), 
                                  size=self.avgGridSize/30.0, pxMode=False)
        data = self.dataHistory[self.historyDisplayIndex]
        self.dataPlot.setData(pos=data, color=(0,1,1,1), 
                              size=self.avgGridSize/30.0, pxMode=False)
        
        self.convergenceLine.setValue(self.historyDisplayIndex)
        dataz = data[:,2]
        self.cost.setText("Cost: " + "{0:.5g}".format(min(dataz)))
            
    def createDataPlot(self):
        initialData = self.dataHistory[0]
        self.dataPlot = gl.GLScatterPlotItem(pos=initialData, color=(0,1,1,1), 
                                             size=self.avgGridSize/30.0, pxMode=False)
        self.prevDataPlot = gl.GLScatterPlotItem(pos=initialData, color=(0,1,1,1), 
                                                 size=self.avgGridSize/30.0, pxMode=False)
        self.w3d.addItem(self.dataPlot)
        self.w3d.addItem(self.prevDataPlot)
        
    def createOptimaPlot(self):
        data = self.costEvaluator.getKnownGlobalMinima()
        if data is not None:
            data = np.array(data)
            minima = gl.GLScatterPlotItem(pos=data, color=(1,0,0,1), 
                                          size=self.avgGridSize/30.0, pxMode=False)
            self.w3d.addItem(minima)
        
    def createGrid(self):
        g = gl.GLGridItem()
        g.scale(2,2,2)
        g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
        self.w3d.addItem(g)
        
    def createConvergencePlot(self):
        if self.convergenceHistory is not None:
            self.convPl = pg.PlotWidget(name='Convergence')
            self.convPl.setLabel('left', 'Cost (Log)')
            self.convPl.setLabel('bottom', 'Iteration')
            self.convPl.plot(self.convergenceHistory)
            # self.convPl.setLogMode(x=False, y=True)
            
            # a line for showing iteration
            self.convergenceLine = pg.InfiniteLine(angle=90, movable=False, pos=0)
            self.convPl.addItem(self.convergenceLine)
            
            grid = pg.GridItem()
            self.convPl.addItem(grid)
            


# if __name__ == '__main__':
    # app = QtGui.QApplication([])
    # xMax = 15
    # yMax = 15
    # costEvaluator = SmithCostEvaluator(0.05, 10)
    
    
    # numData = 10
    # numHistory = 10
    # xData = (np.random.rand(numData, numHistory) - 0.5) * 2 * xMax
    # yData = (np.random.rand(numData, numHistory) - 0.5) * 2 * yMax
    # zData = np.empty((numData, numHistory))
    # for iData in range(numData):
    #     for iHistory in range(numHistory):
    #         x = xData[iData, iHistory]
    #         y = yData[iData, iHistory]
    #         value = np.array([x,y])
    #         z = costEvaluator.getCost(value)
    #         zData[iData, iHistory] = z
    
    # dataHistory = []
    # for i in range(numHistory):
    #     data = np.empty((numData, 3))
    #     data[:,0] = xData[i,:]
    #     data[:,1] = yData[i,:]
    #     data[:,2] = zData[i,:]
    #     dataHistory.append(data)
        
    # visualizer = GeneticVisualizer(costEvaluator=costEvaluator,
    #                                dataHistory=dataHistory)
    # visualizer.setPlotRange(xMax = 15, yMax = 15)
    # visualizer.setResolution(resolution=200)
    # visualizer.visualize()
    # import sys
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()

    
    
