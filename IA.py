import numpy as np
import random
import math
import copy
import time
import matplotlib.pyplot as plt
seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)

alpha=1
class IA:
    def __init__(self,pop_size,dim,limit,delta,beta,pm,colne_num,item):
        '''
        :param pop_size: 个体数目
        :param dim: 变量个数
        :param limit: 变量范围
        :param delta: 相似度阈值
        :param beta:激励度系数
        :param pm:变异概率
        :param colne_num:克隆数目
        '''
        self.pop_size=pop_size
        self.pop=np.random.rand(pop_size,dim)#初始化，每一行是一个个体
        for i in range(dim):
            self.pop[:,i]*=(limit[i][0]-limit[i][1])
            self.pop[:,i]+=limit[i][1]
        self.limit=limit
        self.delta=delta
        self.pm=pm
        self.beta=beta
        self.colne_num=colne_num
        self.item=item#迭代次数
        self.bestvalue=0
        self.bestway=[]
        self.dim=dim

    def Function(self,x):
        fitness=np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            for j in range(self.dim):
                fitness[i]=fitness[i]+x[i,j]**2-10*math.cos(2*math.pi*x[i,j])+10


        # 计算函数适应值
        return fitness



    def cal_density(self):
        '''
        计算浓度
        :return:
        '''
        density=np.zeros(self.pop.shape[0])
        for i in range(self.pop.shape[0]):
            for j in range(self.pop.shape[0]):
                if ((self.pop[i][0]-self.pop[j][0])**2+(self.pop[i][1]-self.pop[j][1])**2)**0.5<self.delta:
                    density[i]+=1
        return density/self.pop.shape[0]

    def cal_simulation(self,simulation,density):
        '''
        计算激励度
        :param simulation: 每个个体的函数值，若求最大值，alpha为正，若求最小值，alpha为负
        :param density:每个个体的浓度
        :return:
        '''
        #减去浓度是为了全局搜索，浓度小的个体也有机会进行克隆操作
        return (alpha*simulation-self.beta*density)

    def mutate(self,x):
        '''
        变异操作，随着代数增加变异范围逐渐减小
        :return:
        '''
        for i in range(self.pop.shape[1]):
            if np.random.rand() <= self.pm:
                # 加上一个随机数产生变异
                x[i] += (np.random.rand() - 0.5) * (self.limit[i][0] - self.limit[i][1]) / (self.item + 1)
                # 边界检测
                if (x[i] > self.limit[i][0] or x[i] < self.limit[i][1]):
                    x[i] = np.random.rand() * (self.limit[i][0] - self.limit[i][1]) + self.limit[i][1]


    def IA_item(self):
        times=[]
        finnal_result=[]
        finnal_value=[]
        for num in range(20):
            result=[]
            start=time.perf_counter()
            self.pop = np.random.rand(self.pop_size, self.dim)  # 初始化，每一行是一个个体
            for i in range(self.dim):
                self.pop[:, i] *= (limit[i][0] - limit[i][1])
                self.pop[:, i] += limit[i][1]
            fitness = self.Function(self.pop)  # 计算适应值
            density = self.cal_density()  # 计算浓度
            simulation = self.cal_simulation(fitness, density)  # 计算激励值
            # 按激励值从大到小排序
            #np.argsort是从小到大排序,加-号使激励值从大到小排序
            index = np.argsort(-simulation)
            self.pop = self.pop[index, :]
            simulation = simulation[index]
            # 开始迭代
            for item in range(self.item):
                # 保存克隆和亲和度最高的个体
                best_a = np.zeros((int(self.pop.shape[0] / 2.0), self.pop.shape[1]))
                best_simluation = np.zeros(int(self.pop.shape[0] / 2))
                new_pop = self.pop.copy()
                # 选出激励值前50%的个体
                for i in range(int(self.pop.shape[0] / 2)):
                    a = new_pop[i, :]
                    # 克隆
                    b = np.tile(a, (self.colne_num, 1))
                    for j in range(self.colne_num):
                        self.mutate(b[j])
                    b_simluation = self.Function(b)
                    # ---------------------------
                    # 最大值在b_simluation前面添加负号
                    if alpha>0:
                        index = np.argsort(-b_simluation)
                    if alpha<0:
                        index=np.argsort(b_simluation)
                    # ----------------------------
                    # 保存克隆里面变异最优的个体
                    best_simluation = b_simluation[index][0]
                    best_a[i, :] = b[index, :][0]
                # 随机生成一半的新个体
                new_pop1 = np.random.rand(int(self.pop.shape[0] / 2), self.pop.shape[1])  # 初始化，每一行是一个个体
                for i in range(self.pop.shape[1]):
                    new_pop1[:, i] *= (limit[i][0] - limit[i][1])
                    new_pop1[:, i] += limit[i][1]
                # 免疫种群与新手种群合并
                self.pop = np.vstack([best_a, new_pop1])
                simulation = self.Function(self.pop)
                if alpha>0:
                    index = np.argsort(-simulation)
                if alpha<0:
                    index = np.argsort(simulation)
                # ----------------------------
                self.bestvalue = simulation[index[0]]
                result.append(self.bestvalue)
                self.bestway = list(self.pop[index[0]])
                density = self.cal_density()
                simulation = self.cal_simulation(simulation, density)
                index = np.argsort(-simulation)
                self.pop = self.pop[index, :]
                simulation = simulation[index]
            end=time.perf_counter()
            times.append(end-start)
            # print("最优解为", self.bestway)
            # print("最优值为", self.bestvalue)

            if alpha >0:
                finnal_result.append(self.bestvalue)
                finnal_value.append(self.bestway)
            if alpha <0:
                finnal_result.append(self.bestvalue)
                finnal_value.append(self.bestway)
            # print('最好的位置：{}'.format(best))
            # print('最大的函数值：{}'.format(self.fitness(best)[1]))
            x = [i for i in range(self.item)]
            print(result)
            plt.plot(x, result)
            plt.show()

        if alpha >0:
            "求极大值"
            print("最优结果", max(finnal_result))
            print("最优解", finnal_value[finnal_result.index(max(finnal_result))])
            print("最差结果", min(finnal_result))
        if alpha <0:
            "求极小值"
            print("最优结果", min(finnal_result))
            print("最优解", finnal_value[finnal_result.index(min(finnal_result))])
            print("最差结果", max(finnal_result))
        # 方差
        print("方差:", np.var(finnal_result))
        print("均值", np.mean(finnal_result))
        print("运行时间,", np.mean(times))
        x = [i for i in range(self.item)]
        plt.plot(x, result)
        plt.show()



limit=[]
for i in range(10):
    limit.append([5.12,-5.12])
ia=IA(100,10,limit,0.2,1,0.7,10,400)
a=ia.IA_item()