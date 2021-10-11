# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 19:26:13 2021

@author: Evan Yu
"""

from dataclasses import dataclass
import numpy as np

    
@dataclass
class DesiredMotion:
    translationX : float
    translationY : float
    rotation : float
    
class State:
    def __init__(self, footState, rotation):
        self.footState = footState
        self.rotation = rotation
        
class Command:
    def __init__(self, footToMove, footPosition, comTranslation, comRotation):
        self.footToMove = footToMove
        self.footPosition = footPosition
        self.comTranslation = comTranslation
        self.comRotation = comRotation
    
def getRotationMatrix(angle):
    rad = np.radians(angle)
    c = np.cos(rad)
    s = np.sin(rad)
    rot = np.array([[c, -s],[s,c]])
    return rot
    