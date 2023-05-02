import copy
import numpy as np
from copy import deepcopy

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

class GA:
    def __init__(self,graph,distmat):
        self.graph=graph
        self.distmat=distmat  # 距离矩阵
        self.dim=distmat.shape[0]  # 维度
        self.og=50  # 种群数目
        self.group=[]  # 种群
        self.path=None
        self.length=0
        self.lengths=[]
        self.pc=0.9  # 交叉概率
        self.pm=0.03  # 变异概率
        self.count=20000  # 迭代数
        self.mc=self.og  # 种群中至少个体数
        self.points=list(zip([self.graph[_,1] for _ in range(self.dim)], [self.graph[_,2] for _ in range(self.dim)],[_ for _ in range(self.dim)]))

    def search(self):
        #self.createOriginalGroup()
        self.newCreate()#生成初始种群方式
        for i in range(self.count):
            self.crossOver_IPMX()  # 交叉
            self.mutate_inversion()  # 变异
            self.select_optimal()  # 选择
            self.path,self.length=self.getOptimal()
            self.lengths.append(self.length)

    # 产生初始种群
    def createOriginalGroup(self):
        origin=[i for i in range(self.dim)]
        for i in range(self.og):
            person=np.random.permutation(origin)
            if person.tostring():
                self.group.append(np.random.permutation(origin))  # 产生随机个体


# 计算两个向量之间的叉积。返回三点之间的关系：    
    def ccw(self,a, b, c):
        return ((b[1] - a[1]) * (c[0] - b[0]))   -    ((c[1] - b[1]) * (b[0] - a[0]) )

# 分别求出后面n-1个点与出发点的斜率，借助sorted，按斜率完成从小到大排序
    def compute(self,next):
        start = self.points[0]  # 第一个点
        if start[0] == next[0]: 
            return 99999
        slope = (start[1] - next[1]) / (start[0] - next[0])  
        return slope

    def Graham_Scan(self,points):
    # # 找到最左边且最下面的点作为出发点，和第一位互换
        Min=999999
        for i in range(len(points)):
        # 寻找最左边的点
            if points[i][0]<Min:
                Min = points[i][0]
                index = i
        # 如果同在最左边，可取y值更小的
            elif points[i][0]==Min:
                if points[i][1]<=points[index][1]:
                    Min = points[i][0]
                    index = i
        # 和第一位互换位置
        temp = points[0]
        points[0] = points[index]
        points[index] = temp
        # 排序：从第二个元素开始，按与第一个元素的斜率排序
        points = points[:1] + sorted(points[1:], key=self.compute)   # 前半部分是出发点；后半部分是经过按斜率排序之后的n-1个坐标点
        #注意： “+”是拼接的含义，不是数值相加
        # 用列表模拟一个栈。（最先加入的是前两个点，前两次while必定不成立，从而将点加进去）
        convex_hull = []
        for p in points:
            while len(convex_hull) > 1 and self.ccw(convex_hull[-2], convex_hull[-1], p) >= 0:
                convex_hull.pop()
            convex_hull.append(p)
        person=[x[2] for x in convex_hull]
        return person

    #优化后的初始种群产生方法
    def newCreate(self):
        stp=self.Graham_Scan(self.points)
        origin=[i for i in range(self.dim)]
        for x in stp:
                origin.remove(x)
        sto=origin
        for i in range(self.og):
            person=stp.copy()#不加.copy()是赋了指针！！！
            #print("st",person)
            origin=sto.copy()
            toto,totp=len(sto),len(stp)
            #print(toto,totp)
            #print(origin)
            while toto>0:
                x=origin[random.randint(0,toto-1)]
                mn,posmn=999999,0
                for j in range(totp):
                    if j<totp-1:Dis=self.distmat[person[j],x]+self.distmat[x,person[j+1]]-self.distmat[person[j],person[j+1]]
                    else:Dis=self.distmat[person[j],x]+self.distmat[x,person[0]]-self.distmat[person[j],person[0]]
                    if Dis<mn:
                        mn=Dis
                        posmn=j
                person.insert(posmn+1,x)
                totp+=1
                origin.remove(x)
                toto-=1
            self.group.append(person)  # 产生随机个体
            #print(len(person),self.dim)
            #print(person)

    # 个体适应值函数
    def fitness(self,status):
        return 1000/self.calDistance(status)

    # 获得种群中的所有个体的适应值和总和
    def groupFitness(self):
        fits=np.zeros(self.og)
        for i in range(self.og):
            fits[i]=self.fitness(self.group[i])
            pass
        return fits

    # 选择个体轮盘赌策略
    def select_roulette(self):
        fits=self.groupFitness()
        temp_group=np.array(self.group)
        new_group_index=np.random.choice(range(self.og),size=self.mc,replace=True,p=fits/fits.sum())
        # 轮盘赌选出相应个体的索引
        new_group=temp_group[new_group_index]  # 选出新个体
        self.group=new_group.tolist()
        self.og=self.mc

    # 精英保留策略
    def select_optimal(self):
        fits=self.groupFitness()
        temp_group=np.array(self.group)
        new_group_index=np.random.choice(self.og,size=self.mc,replace=True,p=fits/fits.sum())
        # 先进行轮盘赌策略筛选个体
        new_group=temp_group[new_group_index]
        new_fits=fits[new_group_index]
        max_fit=fits.argmax()
        min_fit=new_fits.argmin()
        new_group[min_fit]=temp_group[max_fit]  # 最优个体替换
        self.group=new_group.tolist()
        self.og=self.mc

    # 截断选择
    def select_truncation(self):
        fits=self.groupFitness()
        temp_index=[i for i,j in enumerate(fits>=np.average(fits)) if j]
        temp_group=np.array(self.group)
        temp_fits=fits[temp_index]
        temp_group=temp_group[temp_index]  # temp_group中是全部大于平均值的个体
        new_group_index=np.random.choice(len(temp_group),size=self.mc,replace=True,p=temp_fits/temp_fits.sum())
        new_group=temp_group[new_group_index]
        new_fits=temp_fits[new_group_index]
        max_fit=temp_fits.argmax()  # 同时保留精英
        min_fit=new_fits.argmin()
        new_group[min_fit]=temp_group[max_fit]
        self.group=new_group.tolist()
        self.og=self.mc

    def getOptimal(self):
        fits=self.groupFitness()
        return self.group[fits.argmax()],1000/np.max(fits)

    # 交叉
    def crossOver_OX(self):
        temp=self.og
        for i in range(temp // 2):
            if np.random.random()>self.pc:  # 判断是否交叉
                continue
            parent1 = self.group[i]
            parent2 = self.group[i + temp // 2]
            oops = np.random.randint(1, self.dim)
            child1, child2 = list(parent1[:oops].copy()), list(parent2[:oops].copy())
            for j in range(len(parent1)):  # 解决冲突
                if parent1[j] not in child2:
                    child2.append(parent1[j])
                if parent2[j] not in child1:
                    child1.append(parent2[j])
            self.group.append(child1.copy())
            self.group.append(child2.copy())
            self.og+=2

    def crossOver_PMX(self):
        temp=self.og
        for n in range(temp//2):
            if np.random.random()>self.pc:  # 判断是否交叉
                continue
            parent1 = np.array(self.group[n])
            parent2 = np.array(self.group[n + temp // 2])
            child1,child2=copy.deepcopy(parent1),copy.deepcopy(parent2)
            oops0=np.random.randint(0,self.dim-1)
            oops1=np.random.randint(oops0+1,self.dim)
            cross_area=range(oops0,oops1)  # 交叉区索引
            keep_area=np.delete(range(self.dim),cross_area)  # 非交叉区索引
            keep1=parent1[keep_area]
            keep2=parent2[keep_area]
            cross1=parent1[cross_area]
            cross2=parent2[cross_area]
            child1[cross_area],child2[cross_area]=cross2,cross1  # 先对交叉区进行交换
            mapping=[[],[]]  # 映射表
            # 生成映射表
            for i, j in zip(cross1, cross2):
                if j in cross1 and i not in cross2:
                    index = np.argwhere(cross1 == j)[0, 0]
                    value = cross2[index]
                    while value in cross1:
                        index = np.argwhere(cross1 == value)[0, 0]
                        value = cross2[index]
                    mapping[0].append(i)
                    mapping[1].append(value)
                elif j not in cross1 and i not in cross2:
                    mapping[0].append(i)
                    mapping[1].append(j)
            # 根据映射表解决冲突
            for i, j in zip(mapping[0], mapping[1]):
                if i in keep1:
                    keep1[np.argwhere(keep1 == i)[0, 0]] = j
                elif i in keep2:
                    keep2[np.argwhere(keep2 == i)[0, 0]] = j
                if j in keep1:
                    keep1[np.argwhere(keep1 == j)[0, 0]] = i
                elif j in keep2:
                    keep2[np.argwhere(keep2 == j)[0, 0]] = i
            child1[keep_area], child2[keep_area] = keep1, keep2
            self.group.append(child1)
            self.group.append(child2)
            self.og+=2

    def crossOver_IPMX(self):
        temp=self.og
        for n in range(temp//2):
            if np.random.random()>self.pc:  # 判断是否交叉
                continue
            parent1 = np.array(self.group[n])
            parent2 = np.array(self.group[n + temp // 2])
            child1,child2=copy.deepcopy(parent1),copy.deepcopy(parent2)
            oops0=np.random.randint(0,self.dim-1)
            oops1=np.random.randint(oops0+1,self.dim)
            cross_area=range(oops0,oops1)  # 交叉区索引
            keep_area=np.delete(range(self.dim),cross_area)  # 非交叉区索引
            keep1=parent1[keep_area]
            keep2=parent2[keep_area]
            cross1=parent1[cross_area]
            cross2=parent2[cross_area]
            child1[cross_area],child2[cross_area]=cross2,cross1  # 先对交叉区进行交换
            # 生成映射表
            L1=[0 for i in range(self.dim)]
            L2=[0 for i in range(self.dim)]
            dict = {}
            newDict = {}
            for i,j in zip(cross2,cross1):
                dict[i] = j
                L2[i] = 1
                L1[j] = 1
            L1 += L2
            for i in range(self.dim):
                if L1[i] <= 1 and i in dict:
                    pre,cur = i,dict[i]
                    while cur in dict and cur != dict[cur]:
                        cur = dict[cur]
                        if cur == pre: break
                    newDict[pre] = cur
            #根据映射表解决原始后代1的冲突
            for i in keep1:
                if i in newDict:
                    keep1[np.argwhere(keep1 == i)[0, 0]] = newDict[i]
            child1[keep_area], child2[keep_area] = keep1, keep2
            #处理原始后代2的冲突
            F = {}
            for i in range(self.dim):
                F[child1[i]] = parent1[i]
            for i in range(self.dim):
                #if parent2[i] in F:
                    child2[i] = F[parent2[i]]
            #调试查看生成的新子代是否合法
            # Count1=[0 for i in range(self.dim)]
            # Count2=[0 for i in range(self.dim)]
            # tot1,tot2 = 0,0
            # for i in child1: Count1[i] += 1
            # for i in child2: Count2[i] += 1
            # for i in range(self.dim):
            #     if Count1[i]>0:tot1 += 1
            #     if Count2[i]>0:tot2 += 1
            # print(tot1,tot2)
            self.group.append(child1)
            self.group.append(child2)
            self.og+=2

    def mutate_swap(self):
        for i in range(self.og):
            if np.random.random()<self.pm:
                random_site0 = np.random.randint(0, self.dim-1)
                random_site1 = np.random.randint(random_site0,self.dim)
                # 交换两个城市
                self.group[i][random_site0], self.group[i][random_site1] = self.group[i][random_site1], self.group[i][random_site0]

    def mutate_inversion(self):
        for i in range(self.og):
            if np.random.random()<self.pm:
                random_site0 = np.random.randint(0, self.dim-1)
                random_site1 = np.random.randint(random_site0,self.dim)
                # 交换两个城市并将之间的城市倒置
                while random_site0 < random_site1:
                    self.group[i][random_site0],self.group[i][random_site1]=self.group[i][random_site1],self.group[i][random_site0]
                    random_site0+=1
                    random_site1-=1

    # 根据距离矩阵计算距离值
    def calDistance(self,status):
        dis=0
        for i in range(self.dim-1):
            dis+=self.distmat[status[i],status[i+1]]
        dis+=self.distmat[status[0],status[-1]]
        return dis
