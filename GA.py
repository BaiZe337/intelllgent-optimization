import random
import numpy as np
import matplotlib.pyplot as plt
import time
import math
seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)
population_size = 100                      # 种群初始规模
generation_count =2000                     # 遗传代数
num_x=2                                     #变量个数
length_x=10                                  #变量长度
exchange_ratio = 0.8                        # 交叉概率
variation_ratio = 0.01                      # 变异概率
solve_max = True                      # 为True则求解最大值，为False则求解最小值
x_min = [-100,-100]                             # 基因的最小值，即变量x能取到的最小值
x_max = [100,100]                              # 基因的最大值，即变量x能取到的最大值

plt.rcParams['font.sans-serif'] = ['FangSong']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 支持负号显示


def get_chromosome(size, length):
    """
    生成size个长度为length的染色体列表
    :param size: 种群规模规模
    :param length: 染色体长度,长度为变量个数*每个变量的维度
    :return: 二维列表
    """
    population_temp = []
    for i in range(size):
        population_temp.append([random.randint(0, 1) for _ in range(length)])   # 生成长度为length的随机二进制列表，并存放到population_temp列表中
    return population_temp


def get_accuracy(min_, max_, length_x):
    """
    计算搜索精度，取到范围内的每一个值
    :param min_: 变量的最小值
    :param max_: 变量的最大值
    :param length: 变量长度
    :return: 精度
    """
    accuracy=[]
    for i in range(num_x):
        accuracy.append((max_[i] - min_[i]) / (2 ** length_x - 1))
    return accuracy    # 精度计算公式


def chromosome_decode(chromosome_list, min_, accuracy_,num_x,length_x):
    """
    染色体解码
    :param chromosome_list: 二进制染色体列表
    :param min_: 基因的最小值
    :param accuracy_: 精度
    :return: 解码的结果
    """
    #对每一个变量进行解码
    value=[]
    for i in range(num_x):
        b=""
        for j in range(length_x):
            b=b+str(chromosome_list[i*length_x+j])
        b=int(b,2)
        value.append(min_[i] + accuracy_[i] * b)
    return value


def get_fitness(x, solve_flag):
    """
    计算适应度
    :param x: 染色体解码的结果
    :param solve_flag: 求最大值则为True，最小值则为False
    :return: 适应度结果
    """
    fitness=0
    fitness_degree=0
    #计算函数适应值
    fitness=x[0]**2+2*x[1]**2-0.3*math.cos(3*math.pi*x[0]+4*math.pi*x[1])+0.3


    #根据函数适应值求适应度
    if solve_flag==False:
        #取极小值
        if fitness>=0:
            fitness_degree=1.0/(1+fitness)
        else:
            fitness_degree=1+abs(fitness)
    if solve_flag==True:
        #取极大值
        if fitness>=0:
            fitness_degree=1+fitness
        else:
            fitness_degree=1.0/(1+abs(fitness))
    return fitness_degree,fitness


def select(chromosome_list, fitness_list):
    """
    选择(轮盘赌算法)
    :param chromosome_list: 二维列表的种群
    :param fitness_list: 适应度列表
    :return: 选择之后的种群列表
    """
    fitness=np.array(fitness_list)
    idx = np.random.choice(np.arange(len(chromosome_list)), size=len(chromosome_list), replace=True,
                           p=fitness/fitness.sum())
    chromosome=np.array(chromosome_list)
    '''个体选择 end'''
    return list(chromosome[idx])


def exchange(chromosome_list, pc):
    """
    交叉
    :param chromosome_list: 二维列表的种群
    :param pc: 交叉概率
    """
    for i in range(0, len(chromosome_list) - 1, 2):
        if random.uniform(0, 1) < pc:
            c_point = random.randint(0, len(chromosome_list[0]))    # 随机生成交叉点
            '''将第i位和i+1位进行交叉'''
            exchanged_list1 = []
            exchanged_list2 = []
            exchanged_list1.extend(chromosome_list[i][0:c_point])
            exchanged_list1.extend(chromosome_list[i + 1][c_point:len(chromosome_list[i])])
            exchanged_list2.extend(chromosome_list[i + 1][0:c_point])
            exchanged_list2.extend(chromosome_list[i][c_point:len(chromosome_list[i])])


            chromosome_list[i] = exchanged_list1
            chromosome_list[i + 1] = exchanged_list2



def mutation(chromosome_list, pm):
    """
    变异
    :param chromosome_list: 二维列表的种群
    :param pm: 变异概率
    """
    for i in range(len(chromosome_list)):
        if random.uniform(0, 1) < pm:
            m_point = random.randint(0, len(chromosome_list[0]) - 1)    # 随机生成变异点
            chromosome_list[i][m_point] = chromosome_list[i][m_point] ^ 1   # 将该位的值与1异或(即将0置为1,1置为0)


def get_best(fitness_list):
    """
    计算这一代中的最优个体
    :param fitness_list: 适应度列表
    :return: 最优个体的下标
    """
    return fitness_list.index(max(fitness_list))


finnal_result=[]#储存每一次独立运行的最优解
finnal_value=[]
times=[]#储存每一个独立运行的时间
for j in range(20):
    '''独立运行20次'''
    results = []  # 存储每一代的最优解，二维列表
    final_best = []
    gene_length = num_x * length_x
    start = time.perf_counter()
    population = get_chromosome(population_size, gene_length)  # 种群初始化
    for _ in range(generation_count):
        accuracy = get_accuracy(x_min, x_max,length_x)  # 计算搜索精度
        decode_list = [chromosome_decode(individual, x_min, accuracy, num_x, length_x) for individual in
                       population]  # 解码之后的列表
        fit_list = [get_fitness(decode_i, solve_max)[0] for decode_i in decode_list]  # 计算每个个体的适应度
        fitness_list = [get_fitness(decode_i, solve_max)[1] for decode_i in decode_list]  # 计算每个个体的适应值
        results.append(fitness_list[fit_list.index(max(fit_list))])
        final_best.append(decode_list[fit_list.index(max(fit_list))])
        population = select(population.copy(), fit_list)
        exchange(population, exchange_ratio)
        mutation(population, variation_ratio)
    end = time.perf_counter()
    times.append(end-start)
    # print(results)
    if solve_max==True:
        finnal_result.append(max(results))
        finnal_value.append(final_best[results.index(max(results))])
    if solve_max==False:
        finnal_result.append(min(results))
        finnal_value.append(final_best[results.index(min(results))])

if solve_max==True:
    "求极大值"
    print("最优结果",max(finnal_result))
    print("最优解",finnal_value[finnal_result.index(max(finnal_result))])
    print("最差结果",min(finnal_result))
if solve_max==False:
    "求极小值"
    print("最优结果",min(finnal_result))
    print("最优解",finnal_value[finnal_result.index(max(finnal_result))])
    print("最差结果",max(finnal_result))
#方差
print("方差:",np.var(finnal_result))
print("均值",np.mean(finnal_result))
print("平均运行时间,",np.mean(times))
x=[i for i in range(generation_count)]
plt.plot(x,results)
plt.show()

