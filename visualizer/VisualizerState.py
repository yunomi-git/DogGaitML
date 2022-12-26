# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:19:09 2021

@author: Evan Yu
"""

import pyglet
from pyglet import shapes
from model.DogUtil import DogModel, TaskMotion
import numpy as np

from simulation.Dynamics import Dynamics, DynamicsFailureMessage
import util.MathUtil as mu

class VisualizerState():
    def __init__(self):
        self.w = 480
        self.h = 270
        self.dogModel = DogModel()
        
    def drawSingleState(self, currentState, originalCOMInWorld, cost, batch):
        red = (255,0,0)
        yellow = (255, 255, 0)
        green = (0, 255, 0)
        blue = (0,0,255)
        black = (0, 0, 0)
        red = (255,0,0)
        
        thingsToDraw = []
        
        def drawGrid(color): 
            axisShape = shapes.Line(0, self.h,
                                    self.w*2, self.h,
                                    width=1, 
                                    color=color, batch=batch)
            thingsToDraw.append(axisShape)
            axisShape = shapes.Line(self.w, 0,
                                   self.w, self.h*2,
                                   width=1, 
                                   color=color, batch=batch)
            thingsToDraw.append(axisShape)
            
        def drawIdealState(color):
            # draw ideal foot positions green
            #draw circles for boundaries green
            self.dogModel.setState(currentState)
            idealFootState = self.dogModel.getIdealFootStateFromOriginalCom(TaskMotion(0.,0.,0.))
            boundaryRadius = DogModel.maximumFootDistanceFromIdeal
            for i in range(4):
                foot = idealFootState[i,:]
                footShape = shapes.Circle(originalCOMInWorld[0]+foot[0] + self.w, 
                                          originalCOMInWorld[1]+foot[1] + self.h, 
                                          boundaryRadius, color=color, batch=batch)
                footShape.opacity=128
                thingsToDraw.append(footShape)
                
        
        def drawCurrentState(color):
            # draw feet
            currentFootState = currentState.footState
            for i in range(4):
                foot = currentFootState[i,:]
                footShape = shapes.Circle(originalCOMInWorld[0]+foot[0] + self.w,
                                          originalCOMInWorld[1]+foot[1] + self.h, 
                                          5, color=color, batch=batch)
                thingsToDraw.append(footShape)
            # draw com
            comShape = shapes.Star(originalCOMInWorld[0] + self.w, 
                                   originalCOMInWorld[1] + self.h, 
                                   10, 2, num_spikes=4, color=color, batch=batch)
            thingsToDraw.append(comShape)
            # draw rotation
            currentAbsoluteRotation = currentState.absoluteRotation
            direction = mu.getRotationMatrix(currentAbsoluteRotation) @ np.array([100, 0])
            directionShape = shapes.Line(originalCOMInWorld[0] + self.w, 
                                        originalCOMInWorld[1] + self.h,
                                        originalCOMInWorld[0] + self.w + direction[0], 
                                        originalCOMInWorld[1] + self.h + direction[1],
                               width=2, 
                               color=color, batch=batch)
            thingsToDraw.append(directionShape)
        
        def drawCost():
            label = pyglet.text.Label("Cost: " + str(cost),
                          color=(255,255,255,255),
                          font_name='Times New Roman',
                          font_size=15,
                          x=0, y=0,
                          anchor_x='left', anchor_y='bottom', batch=batch)
            thingsToDraw.append(label)
            
            
        
        drawGrid(yellow)
        drawIdealState(green)
        drawCurrentState(black)
        drawCost()
        
        return thingsToDraw
            
        

        
        
    def drawStateWithPrevious(self, originalState, currentState, command, originalCOMInWorld, failureMessage, cost, batch):
        comTranslation = command.comTranslation
        footTranslation = command.footTranslation
        footToMove = command.footToMove
        currentCOMInWorld = originalCOMInWorld + comTranslation        
        thingsToDraw = []
        originalFootState = originalState.footState
        
        red = (255,0,0)
        
        yellow = (255, 255, 0)
        green = (0, 255, 0)
        blue = (0,0,255)
        black = (0, 0, 0)
        red = (255,0,0)

        def drawGrid(color): 
            axisShape = shapes.Line(0, self.h,
                                    self.w*2, self.h,
                                    width=1, 
                                    color=color, batch=batch)
            thingsToDraw.append(axisShape)
            axisShape = shapes.Line(self.w, 0,
                                   self.w, self.h*2,
                                   width=1, 
                                   color=color, batch=batch)
            thingsToDraw.append(axisShape)
        
        def drawIdealState(color):
            # draw ideal foot positions green
            #draw circles for boundaries green
            self.dogModel.setState(originalState)
            idealFootState = self.dogModel.getIdealFootStateFromOriginalCom(command.getTaskMotion())
            boundaryRadius = DogModel.maximumFootDistanceFromIdeal
            for i in range(4):
                foot = idealFootState[i,:]
                footShape = shapes.Circle(originalCOMInWorld[0]+foot[0] + self.w, 
                                          originalCOMInWorld[1]+foot[1] + self.h, 
                                          boundaryRadius, color=color, batch=batch)
                footShape.opacity=128
                thingsToDraw.append(footShape)
        
        def drawOriginalState(color):
            # draw moved last foot position blu
            # draw last com blu
            # draw last orientation arrow blu
            comShape = shapes.Star(originalCOMInWorld[0] + self.w, 
                                   originalCOMInWorld[1] + self.h, 
                                   10, 2, num_spikes=4, color=color, batch=batch)
            thingsToDraw.append(comShape)
            for i in range(4):
                foot = originalFootState[i,:]
                footShape = shapes.Circle(originalCOMInWorld[0]+foot[0] + self.w,
                                          originalCOMInWorld[1]+foot[1] + self.h, 
                                          5, color=color, batch=batch)
                thingsToDraw.append(footShape)
            originalAbsoluteRotation = originalState.absoluteRotation
            direction = mu.getRotationMatrix(originalAbsoluteRotation) @ np.array([100, 0])
            directionShape = shapes.Line(currentCOMInWorld[0] + self.w, 
                                        currentCOMInWorld[1] + self.h,
                                        currentCOMInWorld[0] + self.w + direction[0], 
                                        currentCOMInWorld[1] + self.h + direction[1],
                               width=2, 
                               color=color, batch=batch)
            thingsToDraw.append(directionShape)
        

        def drawNextState(color):
            # draw feet
            currentFootState = currentState.footState
            for i in range(4):
                foot = currentFootState[i,:]
                footShape = shapes.Circle(currentCOMInWorld[0]+foot[0] + self.w,
                                          currentCOMInWorld[1]+foot[1] + self.h, 
                                          5, color=color, batch=batch)
                thingsToDraw.append(footShape)
            # draw com
            comShape = shapes.Star(currentCOMInWorld[0] + self.w, 
                                   currentCOMInWorld[1] + self.h, 
                                   10, 2, num_spikes=4, color=color, batch=batch)
            thingsToDraw.append(comShape)
            # draw rotation
            currentAbsoluteRotation = currentState.absoluteRotation
            direction = mu.getRotationMatrix(currentAbsoluteRotation) @ np.array([100, 0])
            directionShape = shapes.Line(currentCOMInWorld[0] + self.w, 
                                        currentCOMInWorld[1] + self.h,
                                        currentCOMInWorld[0] + self.w + direction[0], 
                                        currentCOMInWorld[1] + self.h + direction[1],
                               width=2, 
                               color=color, batch=batch)
            thingsToDraw.append(directionShape)
        
        def drawCommandState(color):
            #draw support polygon
            triangle =  self.dogModel.getEveryFootExcept(footToMove)
            line = shapes.Line(originalCOMInWorld[0] + triangle[0,0]+ self.w, 
                                originalCOMInWorld[1] + triangle[0,1] + self.h,
                                originalCOMInWorld[0] + triangle[1,0] + self.w, 
                                originalCOMInWorld[1] + triangle[1,1] + self.h,
                               width=1, 
                               color=red, batch=batch)
            thingsToDraw.append(line)
            line = shapes.Line(originalCOMInWorld[0] + triangle[1,0]+ self.w, 
                                originalCOMInWorld[1] + triangle[1,1] + self.h,
                                originalCOMInWorld[0] + triangle[2,0] + self.w, 
                                originalCOMInWorld[1] + triangle[2,1] + self.h,
                               width=1, 
                               color=red, batch=batch)
            thingsToDraw.append(line)
            line = shapes.Line(originalCOMInWorld[0] + triangle[2,0]+ self.w, 
                                originalCOMInWorld[1] + triangle[2,1] + self.h,
                                originalCOMInWorld[0] + triangle[0,0] + self.w, 
                                originalCOMInWorld[1] + triangle[0,1] + self.h,
                               width=1, 
                               color=red, batch=batch)
            thingsToDraw.append(line)
            
            # draw command foot translation arrow red
            movingFoot = originalFootState[footToMove, :]
            commandFoot = movingFoot + footTranslation
            line = shapes.Line(originalCOMInWorld[0] + movingFoot[0]+ self.w, 
                                originalCOMInWorld[1] + movingFoot[1] + self.h,
                                originalCOMInWorld[0] + commandFoot[0] + self.w, 
                                originalCOMInWorld[1] + commandFoot[1] + self.h,
                               width=2, 
                               color=red, batch=batch)
            thingsToDraw.append(line)
            
            # draw command com translation arrow red
            commandCOMInWorld = originalCOMInWorld + comTranslation
            line = shapes.Line(originalCOMInWorld[0] + self.w, 
                                originalCOMInWorld[1] + self.h,
                                commandCOMInWorld[0] + self.w, 
                                commandCOMInWorld[1] + self.h,
                               width=2, 
                               color=red, batch=batch)
            thingsToDraw.append(line)
            
            # com in world
            comShape = shapes.Star(commandCOMInWorld[0] + self.w, 
                                   commandCOMInWorld[1] + self.h, 
                                   10, 2, num_spikes=4, color=color, batch=batch)
            thingsToDraw.append(comShape)
             
            # commanded foot
            footShape = shapes.Circle(originalCOMInWorld[0]+commandFoot[0] + self.w,
                                      originalCOMInWorld[1]+commandFoot[1] + self.h, 
                                      5, color=color, batch=batch)
            thingsToDraw.append(footShape)
            
            commandedAbsoluteRotation = originalState.absoluteRotation + command.comRelativeRotation
            direction = mu.getRotationMatrix(commandedAbsoluteRotation) @ np.array([100, 0])
            directionShape = shapes.Line(currentCOMInWorld[0] + self.w, 
                                        currentCOMInWorld[1] + self.h,
                                        currentCOMInWorld[0] + self.w + direction[0], 
                                        currentCOMInWorld[1] + self.h + direction[1],
                               width=2, 
                               color=color, batch=batch)
            thingsToDraw.append(directionShape)
        
        def drawCost():
            label = pyglet.text.Label("Cost: " + str(cost),
                          color=(255,255,255,255),
                          font_name='Times New Roman',
                          font_size=15,
                          x=0, y=0,
                          anchor_x='left', anchor_y='bottom', batch=batch)
            thingsToDraw.append(label)
        
        def drawFailure():
            label = pyglet.text.Label("Failure",
                      color=(255,0,0,255),
                      font_name='Times New Roman',
                      font_size=15,
                      x=0, y=50,
                      anchor_x='left', anchor_y='bottom', batch=batch)
            thingsToDraw.append(label)

                
            
        
        drawGrid(yellow)
        drawIdealState(green)
        if (failureMessage is None) or (failureMessage.failureHasOccurred()):
            drawCommandState(red)
            drawOriginalState(black)
            drawFailure();
        else:
            drawCommandState(red)
            drawOriginalState(blue)
            drawNextState(black)
        drawCost()
        
        
        return thingsToDraw