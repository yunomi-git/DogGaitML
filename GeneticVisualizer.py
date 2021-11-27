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
from CostEvaluator import CostEvaluator

class CostEvaluator2D(CostEvaluator):
    def __init__(self):
        pass
    
    def getCostMesh(self, xArr, yArr):
        xLength = xArr.size
        yLength = yArr.size
        z = np.empty((xLength, yLength))
        for ix in range(xLength):
            for iy in range(yLength):
                value = np.array([xArr[ix], yArr[iy]])
                z[ix,iy] = self.getCost(value)
        return z
        # 
        # x = x.reshape(xLength, 1)
        # yLength = y.size
        # y = y.reshape(1, yLength)    
        # return self.getCost(x, y)
    
class ParaboloidCostEvaluator(CostEvaluator2D):
    def __init__(self, a, b, c, d, e):
        self.a = a;
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        
    def getCost(self, value):
        x = value[0]
        y = value[1]
        return (self.a * x * x + self.b * y * y + 
                self.c * x + self.d * y +
                self.e * x * y)
    
class SmithCostEvaluator(CostEvaluator2D):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def getCost(self, value):
        x = value[0]
        y = value[1]
        return self.a * (x * (np.cos(self.b * x * y + 1/y) + x) + y * y) + 1



    
class GeneticVisualizer():
    def __init__(self, costEvaluator, dataHistory):
       self.setDefaultOptions()
       self.costEvaluator = costEvaluator
       self.historyDisplayIndex = 0
       self.setDataHistory(dataHistory)
       
       # self.createEverything()
       
    def setDefaultOptions(self):
        self.setPlotRange(xMax = 15, yMax = 15)
        self.setResolution(200)
       
    #bug - this won't update if called after initialization.
    # consider letting user call createEverything manually
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
        self.w3d.setCameraPosition(distance=50)
        
        self.createCostSurface()
        self.createDataPlot()
        self.createGrid()
        self.createButtons()
        
        self.wMain.addWidget(self.w3d, row = 1, col = 0, colspan = 2)
        self.wMain.show()
        self.wMain.resize(800,500)
        
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
        
        self.wMain.addWidget(self.prevBtn, row=0, col=0)
        self.wMain.addWidget(self.nextBtn, row=0, col=1)
    
    def nextButton(self):
        self.incrementHistory(1)
    def prevButton(self):
        self.incrementHistory(-1)
        
    def incrementHistory(self, i):        
        self.historyDisplayIndex += i
        if self.historyDisplayIndex > (self.numHistory - 1):
            self.historyDisplayIndex = self.numHistory - 1
        if self.historyDisplayIndex < 0:
            self.historyDisplayIndex = 0
        prevhistoryDisplayIndex = self.historyDisplayIndex - 1
        if prevhistoryDisplayIndex < 0:
            prevhistoryDisplayIndex = 0
        prevData = self.dataHistory[prevhistoryDisplayIndex]
        self.prevDataPlot.setData(pos=prevData, color=(0,1,0,0.5), 
                                  size=self.avgGridSize/30.0, pxMode=False)
        data = self.dataHistory[self.historyDisplayIndex]
        self.dataPlot.setData(pos=data, color=(0,1,1,1), 
                              size=self.avgGridSize/30.0, pxMode=False)
            
    def createDataPlot(self):
        initialData = self.dataHistory[0]
        self.dataPlot = gl.GLScatterPlotItem(pos=initialData, color=(0,1,1,1), 
                                             size=self.avgGridSize/30.0, pxMode=False)
        self.prevDataPlot = gl.GLScatterPlotItem(pos=initialData, color=(0,1,1,1), 
                                                 size=self.avgGridSize/30.0, pxMode=False)
        self.w3d.addItem(self.dataPlot)
        self.w3d.addItem(self.prevDataPlot)
        
    def createGrid(self):
        g = gl.GLGridItem()
        g.scale(2,2,1)
        g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
        self.w3d.addItem(g)


if __name__ == '__main__':
    app = QtGui.QApplication([])
    xMax = 15
    yMax = 15
    costEvaluator = SmithCostEvaluator(0.05, 10)
    
    
    numData = 10
    numHistory = 10
    xData = (np.random.rand(numData, numHistory) - 0.5) * 2 * xMax
    yData = (np.random.rand(numData, numHistory) - 0.5) * 2 * yMax
    zData = np.empty((numData, numHistory))
    for iData in range(numData):
        for iHistory in range(numHistory):
            x = xData[iData, iHistory]
            y = yData[iData, iHistory]
            value = np.array([x,y])
            z = costEvaluator.getCost(value)
            zData[iData, iHistory] = z
    
    dataHistory = []
    for i in range(numHistory):
        data = np.empty((numData, 3))
        data[:,0] = xData[i,:]
        data[:,1] = yData[i,:]
        data[:,2] = zData[i,:]
        dataHistory.append(data)
        
    visualizer = GeneticVisualizer(costEvaluator=costEvaluator,
                                   dataHistory=dataHistory)
    visualizer.setPlotRange(xMax = 15, yMax = 15)
    visualizer.setResolution(resolution=200)
    visualizer.visualize()
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

    
    
