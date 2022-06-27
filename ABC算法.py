import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from random import *
import math
import time
seed(1,1)#设置随机数种子
xmax=[100,100]#设置维度取值范围
xmin=[-100,-100]
# for i in range(10):
#     xmax.append(30)
#     xmin.append(-30)


class Bee:
    def __init__(self,dimension):
        self.dimension=dimension#变量个数
        self.fitdegreevalue=0#计算适应度
        self.fitvalue=0
        self.limit=0#限制一个蜜源的迭代次数
        self.x=np.zeros((1,dimension))
        for i in range(dimension):
            self.x[0,i]=random()*(xmax[i]-xmin[i])+xmin[i]
class BeeQun:
    def __init__(self,sn,dimension,mcn,limit,solve_max):
        '''
        :param sn: 雇佣峰的数量，蜜源的数量，观察峰的数量相等
        :param dimension:可行解的维数
        :param mcn:终止代数
        :param limit:为防止算法陷入局部最优，蜜源最大改进次数
        :param k:聚类数
        '''
        self.miyuan=[]
        self.employbee=[]
        self.onlookerbee=[]
        self.selectprob=[]#蜜源被选择的概率
        self.sn=sn
        self.solve_max=solve_max
        self.demension=dimension
        self.limit=limit
        self.mcn=mcn
        for i in range(self.sn):
            #蜜源初始化
            self.miyuan.append(Bee(self.demension))#初始化蜜源
            beefunvalue=self.CalculateFitValue(self.miyuan[i])[0]#计算蜜源的适应度
            self.miyuan[i].fitdegreevalue=beefunvalue
            #雇佣峰初始化
            self.employbee.append(Bee(self.demension))#初始化雇佣蜂
            beefunvalue=self.CalculateFitValue(self.employbee[i])[0]#计算蜜源的适应度
            self.employbee[i].fitdegreevalue=beefunvalue
            #观察峰初始化
            self.onlookerbee.append(Bee(self.demension))#初始化观察蜂
            beefunvalue=self.CalculateFitValue(self.onlookerbee[i])[0]#计算蜜源的适应度
            self.onlookerbee[i].fitdegreevalue=beefunvalue
            #概率初始化
            self.selectprob.append(0)
        self.bestmiyuan=Bee(self.demension)


    def CalculateFitValue(self,Bee):
        '''
        计算适应值的函数
        :param Bee: 蜜源
        :return: 适应值
        '''
        fitness=0
        fitness_degree=0

        fitness = Bee.x[0,0] ** 2 + 2 * Bee.x[0,1] ** 2 - 0.3 * math.cos(3 * math.pi * Bee.x[0,0] + 4 * math.pi * Bee.x[0,1]) + 0.3

        # 根据函数适应值求适应度
        if self.solve_max == False:
            # 取极小值
            if fitness >= 0:
                fitness_degree = 1.0 / (1 + fitness)
            else:
                fitness_degree = 1 + abs(fitness)
        if self.solve_max == True:
            # 取极大值
            if fitness >= 0:
                fitness_degree = 1 + fitness
            else:
                fitness_degree = 1.0 / (1 + abs(fitness))
        return fitness_degree, fitness

    def EmployedBeeOperate(self):
        '''
        雇佣峰的操作函数
        :return:
        '''
        for i in range(self.sn):
            for j in range(self.demension):
                #利用先前的蜜源信息寻找新的蜜源
                self.employbee[i].x[0,j]=self.miyuan[i].x[0,j]
            jie=randint(0,self.demension-1)
            while True :
                k=randint(0,self.sn-1)
                if k!=i:
                    break

            #搜索旧蜜源附近的新蜜源
            self.employbee[i].x[0,jie]=self.miyuan[i].x[0,jie]+uniform(-1,1)*\
                                       (self.miyuan[i].x[0,jie]-self.miyuan[k].x[0,jie])

            #修正
            if self.employbee[i].x[0,jie]>xmax[jie]:
                self.employbee[i].x[0,jie]=xmax[jie]
            elif self.employbee[i].x[0, jie] < xmin[jie]:
                self.employbee[i].x[0, jie] = xmin[jie]
            self.employbee[i].fitdegreevalue=self.CalculateFitValue(self.employbee[i])[0]

            #是否找到更好的蜜源,更新蜜源;如果没找到limit+1
            if(self.employbee[i].fitdegreevalue>self.miyuan[i].fitdegreevalue):
                for k in range(self.demension):
                    self.miyuan[i].x[0,k]=self.employbee[i].x[0,k]
                self.miyuan[i].limit=0
                self.miyuan[i].fitdegreevalue=self.CalculateFitValue(self.miyuan[i])[0]
                self.miyuan[i].fitvalue=self.CalculateFitValue(self.miyuan[i])[1]
            else:
                self.miyuan[i].limit=self.miyuan[i].limit+1


    def CalculateProb(self):
        '''
        计算蜜源被选择的概率
        :return:
        '''
        sumvalue=0
        for i in range(self.sn):
            sumvalue=sumvalue+self.CalculateFitValue(self.miyuan[i])[0]
        for i in range(self.sn):
            self.selectprob[i]=self.miyuan[i].fitdegreevalue/sumvalue

    def onlookerBeeOperate(self):
        '''
        观察峰的操作
        :return:
        '''
        m=0
        while m<self.sn:#为所有的观察峰按照概率选择蜜源并搜索新蜜源
            choose_prob=random()
            for i in range(self.sn):
                choose_prob = choose_prob - self.selectprob[i]
                if choose_prob<=0:
                    jie = randint(0,self.demension-1)
                    while True:
                        k = randint(0, self.sn-1)
                        if k != i:
                            break
                    for j in range(self.demension):
                        # 利用先前的蜜源信息寻找新的蜜源
                        self.onlookerbee[m].x[0, j] = self.miyuan[i].x[0, j]
                    # 搜索旧蜜源附近的新蜜源
                    self.onlookerbee[m].x[0, jie] = self.miyuan[i].x[0, jie] + uniform(-1, 1) * \
                                                  (self.miyuan[i].x[0, jie] - self.miyuan[k].x[0, jie])

                    # 修正
                    if self.onlookerbee[m].x[0, jie] > xmax[jie]:
                        self.onlookerbee[m].x[0, jie] = xmax[jie]
                    elif self.onlookerbee[m].x[0, jie] < xmin[jie]:
                        self.onlookerbee[m].x[0, jie] = xmin[jie]
                    self.onlookerbee[m].fitdegreevalue = self.CalculateFitValue(self.onlookerbee[m])[0]

                    # 是否找到更好的蜜源,更新蜜源;如果没找到limit+1
                    if (self.onlookerbee[m].fitdegreevalue > self.miyuan[i].fitdegreevalue):
                        for k in range(self.demension):
                            self.miyuan[i].x[0, k] = self.onlookerbee[m].x[0, k]
                        self.miyuan[i].limit = 0
                        self.miyuan[i].fitdegreevalue = self.CalculateFitValue(self.miyuan[i])[0]
                    else:
                        self.miyuan[i].limit = self.miyuan[i].limit + 1
                    m=m+1;
                    break

    def ScoutBeeOperate(self):
        '''
        侦查蜂的操作
        :return:
        '''
        #如果蜜源超过一定次数没有找到更优位置，则放弃该蜜源，重新生成蜜源
        for i in range(self.sn):
            if self.miyuan[i].limit>self.limit:
                for j in range(self.demension):
                    self.miyuan[i].x[0,j]=random()*(xmax[j]-xmin[j])+xmin[j]
                self.miyuan[i].limit=0
                self.miyuan[i].fitdegreevalue=self.CalculateFitValue(self.miyuan[i])[0]

    def Savebestmiyuan(self):
        for i in range(self.sn):
            if self.CalculateFitValue(self.miyuan[i])[0]>self.CalculateFitValue(self.bestmiyuan)[0]:
                for j in range(self.demension):
                    self.bestmiyuan.x[0,j]=self.miyuan[i].x[0,j]
                    self.bestmiyuan.fitdegreevalue=self.CalculateFitValue(self.bestmiyuan)[0]
                    self.bestmiyuan.fitvalue=self.CalculateFitValue(self.bestmiyuan)[1]

    def IABC(self):
        times=[]
        finnal_result=[]
        finnal_value=[]
        for j in range(20):
            start=time.perf_counter()
            result=[]
            self.Savebestmiyuan()
            for i in range(self.mcn):
                self.EmployedBeeOperate()
                self.CalculateProb()
                self.onlookerBeeOperate()
                # self.Savebestmiyuan()
                self.ScoutBeeOperate()
                self.Savebestmiyuan()
                result.append(self.bestmiyuan.fitvalue)
                # print("第"+str(i)+"次迭代的最优解为"+str(self.bestmiyuan.fitdegreevalue))
            end=time.perf_counter()
            times.append(end-start)

            if self.solve_max==True:
                finnal_result.append(max(result))
                finnal_value.append(self.bestmiyuan.x)
            if self.solve_max==False:
                finnal_result.append(min(result))
                finnal_value.append(self.bestmiyuan.x)
            # print('最好的位置：{}'.format(best))
            # print('最大的函数值：{}'.format(self.fitness(best)[1]))


        if self.solve_max == True:
            "求极大值"
            print("最优结果", max(finnal_result))
            print("最优解",finnal_value[finnal_result.index(max(finnal_result))])
            print("最差结果", min(finnal_result))
        if self.solve_max == False:
            "求极小值"
            print("最优结果", min(finnal_result))
            print("最优解",finnal_value[finnal_result.index(min(finnal_result))])
            print("最差结果", max(finnal_result))
        # 方差
        print("方差:", np.var(finnal_result))
        print("均值", np.mean(finnal_result))
        print("运行时间,", np.mean(times))
        x = [i for i in range(self.mcn)]
        plt.plot(x, result)
        plt.show()

        return self.bestmiyuan.x

b=BeeQun(100,2,2000,20,True)
b.IABC()



