# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 21:54:15 2021

@author: Evan Yu
"""
import numpy as np
import matplotlib.pyplot as plt;

# class Polygon2D():
#     def __init__(self, pointsList):
#         self.pointsList = pointsList #2d numpy array preferred of nx2
        
#     def getCentroid(self):
#         pass
    
#     def isPointEnclosed(self, point):
#         pass
    
class Triangle2D():
    def __init__(self, pointsList):
        self.pointsList = pointsList #2d numpy array preferred of nx2
        
    def getCentroid(self):
        centroid = np.zeros([2])
        for i in range(3):
            centroid += self.pointsList[i,:]/3
        return centroid
    
    # from https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
    def isPointEnclosed(self, point):
        v1 = self.pointsList[0,:]
        v2 = self.pointsList[1,:]
        v3 = self.pointsList[2,:]
        
        d1 = self.getSign(point, v1, v2)
        d2 = self.getSign(point, v2, v3)
        d3 = self.getSign(point, v3, v1)

        hasNeg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        hasPos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (hasNeg and hasPos)        
    
    def getSign(self, p1,p2,p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    
    
    
    
    
def testTriangle():
    trianglePoints = np.array([[1,1],[2,9],[9,6]])
    triangle = Triangle2D(trianglePoints);
    
    fig, ax = plt.subplots()
    
    xRange = np.linspace(0,10,100);
    yRange = np.linspace(0,10,100);
    
    enclosedx = []
    enclosedy = []
    excludedx = []
    excludedy = []
    
    for x in xRange:
        for y in yRange:
            p = np.array([x, y])
            if triangle.isPointEnclosed(p):
                enclosedx.append(x)
                enclosedy.append(y)
            else:
                excludedx.append(x)
                excludedy.append(y)
                
    ax.scatter(enclosedx, enclosedy, c='g')
    ax.scatter(excludedx, excludedy, c='b')
    
    centroid = triangle.getCentroid()
    ax.scatter(centroid[0], centroid[1], c='r')
    
                    
def main():
    testTriangle()
    
    
if __name__ == "__main__":
    main()