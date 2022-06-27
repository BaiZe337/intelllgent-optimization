# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as mpl
import time
seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)
mpl.rcParams['font.sans-serif'] = ['SimHei']
class PSO:
    def __init__(self, dimension, time, size, low, up, v_low, v_high,solve_max):
        # 初始化
        self.dimension = dimension  # 变量个数
        self.time = time  # 迭代的代数
        self.size = size  # 种群大小
        self.bound = []  # 变量的约束范围
        self.bound.append(low)
        self.bound.append(up)
        self.v_low = v_low
        self.v_high = v_high
        self.x = np.zeros((self.size, self.dimension))  # 所有粒子的位置
        self.v = np.zeros((self.size, self.dimension))  # 所有粒子的速度
        self.p_best = np.zeros((self.size, self.dimension))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.dimension))[0]  # 全局最优的位置
        self.solve_max=solve_max

        # 初始化第0代初始全局最优解
        if self.solve_max==True:
            temp = -float("inf")
        if self.solve_max==False:
            temp=float("inf")
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.v[i][j] = random.uniform(self.v_low, self.v_high)
            self.p_best[i] = self.x[i]  # 储存最优的个体
            fit = self.fitness(self.p_best[i])
            # 做出修改
            if self.solve_max==True:
                if fit > temp:
                    self.g_best = self.p_best[i]
                    temp = fit
            if self.solve_max==False:
                if fit <temp:
                    self.g_best=self.p_best[i]
                    temp=fit


    def fitness(self, x):
        """
        个体适应值计算
        """
        fitness=0
        # 计算函数适应值
        fitness=x[0]**2+2*x[1]**2-0.3*math.cos(3*math.pi*x[0]+4*math.pi*x[1])+0.3
        return fitness

    def update(self, size):
        c1 = 2.0  # 学习因子
        c2 = 2.0
        w = 0.8  # 自身权重因子
        for i in range(size):
            # 更新速度(核心公式)
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # 速度限制
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low:
                    self.v[i][j] = self.v_low
                if self.v[i][j] > self.v_high:
                    self.v[i][j] = self.v_high

            # 更新位置
            self.x[i] = self.x[i] + self.v[i]
            # 位置限制
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
            # 更新p_best和g_best
            if self.solve_max==True:
                if self.fitness(self.x[i]) > self.fitness(self.p_best[i]):
                    self.p_best[i] = self.x[i]
                if self.fitness(self.x[i]) > self.fitness(self.g_best):
                    self.g_best = self.x[i]
            if self.solve_max==False:
                if self.fitness(self.x[i]) < self.fitness(self.p_best[i]):
                    self.p_best[i] = self.x[i]
                if self.fitness(self.x[i]) < self.fitness(self.g_best):
                    self.g_best = self.x[i]


    def pso(self):
        finnal_result=[]
        finnal_value=[]
        times=[]
        for i in range(20):
            best = []
            start = time.perf_counter()
            self.final_best = np.zeros((1, self.dimension))[0]
            for i in range(self.dimension):
                self.final_best[i] = random.uniform(self.bound[0][i], self.bound[1][i])
            for gen in range(self.time):
                self.update(self.size)
                if self.solve_max == True:
                    if self.fitness(self.g_best) > self.fitness(self.final_best):
                        self.final_best = self.g_best.copy()
                if self.solve_max == False:
                    if self.fitness(self.g_best) < self.fitness(self.final_best):
                        self.final_best = self.g_best.copy()
                temp = self.fitness(self.final_best)
                best.append(temp)
            end = time.perf_counter()
            if self.solve_max==True:
                finnal_result.append(max(best))
                finnal_value.append(self.final_best)
            if self.solve_max==False:
                finnal_result.append(min(best))
                finnal_value.append(self.final_best)
            times.append(end-start)


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
        x = [i for i in range(self.time)]
        plt.plot(x,best)
        plt.show()
low = [-100,-100]
up = [100,100]
# for i in range(2):
#     low.append(-30)
#     up.append(30)
pso=PSO(2,2000,100,low,up,-1,1,True)
pso.pso()