import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math
seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)
class ACO:
    def __init__(self, item,pop_size,num_x,min,max,solve_max):
        """
        Ant Colony Optimization
        parameter: a list type, like [NGEN, pop_size, var_num_min, var_num_max]
        """
        # 初始化
        self.item = item  # 迭代的代数
        self.pop_size = pop_size  # 种群大小
        self.var_num = num_x # 变量个数
        self.bound = []  # 变量的约束范围
        self.bound.append(min)
        self.bound.append(max)
        self.solve_max=solve_max

        self.pop_x = np.zeros((self.pop_size, self.var_num))  # 所有蚂蚁的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局蚂蚁最优的位置

        # 初始化第0代初始全局最优解
        temp = -1
        if self.solve_max==True:
            temp = -float("inf")
        if self.solve_max==False:
            temp=float("inf")
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = np.random.uniform(self.bound[0][j], self.bound[1][j])
            fit = self.fitness(self.pop_x[i])[1]
            if self.solve_max==True:
                if fit > temp:
                    self.g_best = self.pop_x[i]
                    temp = fit
            if self.solve_max==False:
                if fit <temp:
                    self.g_best=self.pop_x[i]
                    temp=fit


    def fitness(self, x):
        """
        个体适应值计算
        """
        fitness=0
        fitness_degree=0
        # 计算函数适应值
        fitness=x[0]**2+2*x[1]**2-0.3*math.cos(3*math.pi*x[0]+4*math.pi*x[1])+0.3

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

    def update_operator(self, gen, t, t_min):
        """
        更新算子：根据概率更新下一时刻的位置
        """
        rou = 0.8  # 信息素挥发系数
        Q = 1  # 信息释放总量
        lamda = 1 / gen
        pi = np.zeros(self.pop_size)

        for i in range(self.pop_size):
            for j in range(self.var_num):
                pi[i] = (t[i]-t_min) / t_min
                # 更新位置
                #如果信息素多，则蚂蚁在该位置附近进行局部搜索
                if pi[i] < np.random.uniform(0, 1):
                    self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * lamda
                else:
                    #如果信息素少，则重新生成蚂蚁的位置，全局搜索
                    self.pop_x[i][j] = self.pop_x[i][j] + np.random.uniform(-1, 1) * (
                            self.bound[1][j] - self.bound[0][j]) / 2
                # 越界保护
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]
            # 更新信息素的值
            t[i] = (1 - rou) * t[i] + Q * self.fitness(self.pop_x[i])[0]
            # 更新全局最优值
            if self.fitness(self.pop_x[i])[0] > self.fitness(self.g_best)[0]:
                self.g_best = self.pop_x[i].copy()
        t_min = np.min(t)
        return t_min, t

    def main(self):
        times=[]
        finnal_result=[]
        finnal_value=[]
        for i in range(20):
            popobj = []
            result = []
            best = np.zeros((1, self.var_num))[0]
            best=self.g_best
            start = time.perf_counter()
            for gen in range(1, self.item + 1):
                if gen == 1:
                    # 对于第一代，适应值越大（越小），信息素越多
                    fit_degree = [self.fitness(x)[0] for x in self.pop_x]
                    fitness = [self.fitness(x)[1] for x in self.pop_x]
                    result.append(fitness[fit_degree.index(max(fit_degree))])
                    tmin, t = self.update_operator(gen, np.array(fit_degree),
                                                   np.min(np.array(fit_degree)))
                else:
                    fit_degree = [self.fitness(x)[0] for x in self.pop_x]
                    fitness = [self.fitness(x)[1] for x in self.pop_x]
                    result.append(fitness[fit_degree.index(max(fit_degree))])
                    tmin, t = self.update_operator(gen, t, tmin)
                popobj.append(self.fitness(self.g_best)[1])
                if self.fitness(self.g_best)[0] < self.fitness(best)[0]:
                    best = self.g_best.copy()
            end = time.perf_counter()
            times.append(end-start)
            if self.solve_max==True:
                finnal_result.append(max(result))
                finnal_value.append(best)
            if self.solve_max==False:
                finnal_result.append(min(result))
                finnal_value.append(best)
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

        x=[i for i in range(self.item)]
        plt.plot(x,popobj)
        plt.show()

low = [-100,-100]
up = [100,100]
# for i in range(10):
#     low.append(-30)
#     up.append(30)
aco = ACO(2000,100,2,low,up,True)
aco.main()