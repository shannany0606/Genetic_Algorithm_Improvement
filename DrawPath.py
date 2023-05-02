import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib.animation import FuncAnimation

# 绘制路径地图
def drawPath(graph,path:numpy.ndarray,lengths,count):
    plt.figure(1)
    plt.subplot(1,2,1)
    x,y=[],[]
    for i in path:
        x.append(graph[i,1])
        y.append(graph[i,2])
    x.append(graph[path[0],1])
    y.append(graph[path[0], 2])
    plt.plot(x,y,color='r')
    plt.scatter(x,y,color='b')
    plt.subplot(1,2,2)
    plt.plot(np.arange(count),lengths)
    plt.show()
    pass

def showPath(graph,paths:list):
    plt.ion()
    plt.figure(2)
    x, y = [], []
    for path in paths:
        for i in path:
            x.append(graph[i,1])
            y.append(graph[i,2])
        x.append(graph[path[0],1])
        y.append(graph[path[0], 2])
        plt.plot(x,y,color='r')
        plt.scatter(x,y,color='b')
        plt.pause(0.02)
        x.clear()
        y.clear()
        plt.clf()

