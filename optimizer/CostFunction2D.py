# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:58:24 2021

@author: Evan Yu
"""

from abc import ABC, abstractmethod
import numpy as np
from optimizer.CostEvaluator import CostEvaluator

class CostEvaluator2D(CostEvaluator):
    def __init__(self):
        pass
    
    @abstractmethod
    def getDefaultRange(self):
        pass
    
    @abstractmethod
    def getKnownGlobalMinima(self):
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
    
class ParaboloidCostEvaluator(CostEvaluator2D):
    def __init__(self, a, b, c, d, e):
        self.a = a;
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        
    def getDefaultRange(self):
        return (15, 15)
    
    def getKnownGlobalMinima(self):
        return None
        
    def getCost(self, value):
        x = value[0]
        y = value[1]
        return (self.a * x * x + self.b * y * y + 
                self.c * x + self.d * y +
                self.e * x * y)
    
class SmithCostEvaluator(CostEvaluator2D):
    def __init__(self, a=0.05, b=10):
        self.a = a
        self.b = b
        
    def getDefaultRange(self):
        return (15, 15)
    
    def getKnownGlobalMinima(self):
        return None
    
    def getCost(self, value):
        x = value[0]
        y = value[1]
        return self.a * (x * (np.cos(self.b * x * y + 1/y) + x) + y * y) - 5
    
class SixHumpCamelCostEvaluator(CostEvaluator2D):
    def __init__(self):
        pass
        
    def getDefaultRange(self):
        return (2.5, 1.5)
    
    def getKnownGlobalMinima(self):
        return [np.array([0.0898, -0.7126, -0.10316 + 0.2]),
                np.array([0.0898, 0.7126, -0.10316 + 0.2])]
    
    def getCost(self, value):
        x = value[0]
        y = value[1]
        return ((4. - 2.1 * x*x + np.power(x, 4) / 3.0) * x * x + x * y + (-4. + 4. * y*y) * y*y + 2) * 0.1
    
class AckleyCostEvaluator(CostEvaluator2D):
    def __init__(self, a=20, b=0.2, c=6.28318530):
        self.a = a
        self.b = b
        self.c = c
        
    def getDefaultRange(self):
        return (32.768, 32.768)
    
    def getKnownGlobalMinima(self):
        return [np.array([0, 0, 0])]
    
    def getCost(self, value):
        x = value[0]
        y = value[1]
        return (-self.a * np.exp(-self.b * np.sqrt(0.5 * (x*x+y*y))) 
                - np.exp(0.5 * (np.cos(self.c * x) + np.cos(self.c * y)))
                + self.a + np.exp(1))
    
class EggHolderCostEvaluator(CostEvaluator2D):
    def __init__(self):
        pass
        
    def getDefaultRange(self):
        return (512, 512)
    
    def getKnownGlobalMinima(self):
        return [np.array([512, 404.2319, -95.96407])]
    
    def getCost(self, value):
        x = value[0]
        y = value[1]
        return (-(y + 47) * np.sin(np.sqrt(np.abs(y + x / 2.0 + 47)))
                -x * np.sin(np.sqrt(np.abs(x - y - 47)))) * 0.1


