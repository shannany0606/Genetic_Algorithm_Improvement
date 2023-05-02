import time
import numpy as np
from GA import GA
import DrawPath

fileName=['ch130.tsp','gr120.tsp','lin105.tsp','rd100.tsp']
optName=['ch130.opt.tour','gr120.opt.tour','lin105.opt.tour','rd100.opt.tour']

def getOriginalData(choose):
    with open('data/'+fileName[choose]) as fp:
        lines=fp.readlines()
        dim=int(lines[3].split(' ')[-1])
        graph=np.zeros((dim,3))  # 0:city index 1:x 2:y
        distmat=np.zeros((dim,dim))
        for i in range(dim):
            for j,pox in enumerate(filter(lambda x: x and x.strip(),lines[i+6].split(' '))):
                graph[i][j]=float(pox)
        for i in range(dim):
            for j in range(i,dim):
                if i==j:
                    distmat[i][j]=float('inf')
                else:
                    distmat[i][j]=distmat[j][i]=np.linalg.norm(graph[i,1:]-graph[j,1:])
        return graph,distmat

def getMinDis(choose,distmat):
    with open('data/'+optName[choose]) as fp:
        lines=fp.readlines()
        dim=int(lines[3].split(' ')[-1])
        minDis=0
        pre=1
        cur=0
        for i in range(1,dim):
            cur=eval(lines[i+5])
            #print(pre,cur)
            minDis+=distmat[pre-1][cur-1]
            pre=cur
        return minDis+distmat[cur-1][0]
        
def start():
    print("1.ch130")
    print("2.gr120")
    print("3.lin105")
    print("4.rd100")
    print("请输入您选择的测试集序号",end=":")
    choose=int(input())
    #choose=1
    choose-=1
    if choose>=len(fileName):
        return
    graph,distmat=getOriginalData(choose)
    alg=GA(graph,distmat)
    start=time.process_time()
    alg.search()
    end=time.process_time()
    minDis=getMinDis(choose,distmat)
    print(f"当前路径长:{alg.length}")
    print(f"最优路径长:{minDis}")
    print(f"搜索时间:{end-start}s")
    DrawPath.drawPath(graph, alg.path,alg.lengths,alg.count)
