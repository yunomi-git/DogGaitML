# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 23:09:47 2021

@author: Evan Yu
"""
from abc import ABC, abstractmethod

class CostEvaluator(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def getCost(self, value):
        pass
    
class ParabolicCostEvaluator(CostEvaluator):
    def __init__(self, a, b):
        self.a = a;
        self.b = b
        
    def getCost(self, value):
        x = value[0];
        return (self.a * x * x + self.b * x)
    
class ParaboloidCostEvaluator(CostEvaluator):
    def __init__(self, a, b):
        self.a = a;
        self.b = b
        
    def getCost(self, value):
        x = value[0];
        y = value[1];
        return (self.a * x * x + self.b * y * y)