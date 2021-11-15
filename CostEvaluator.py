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
    
