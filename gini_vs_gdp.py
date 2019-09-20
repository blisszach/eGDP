# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 12:29:02 2019

@author: yankee doodle
"""
import random
import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import itertools

N = 4
M = 20

min_i = 6
max_i = M

opp_type = 'sin'

def O(v):
    if v == 0:
        return 0
    if v < min_i or v > max_i:
        return 0
    else:
        if opp_type == 'uniform':
            return 1
        if opp_type == 'sin':
            return math.sin((v-min_i)*math.pi/(max_i-min_i))
        if opp_type == 'exp':
            return math.exp(v-min_i)
        if opp_type == 'negexp':
            return math.exp(-(v-min_i))

    
def P(v,w):
    if v > w or w == 0 or v == 0:
        return 0
    if v > max_i or v < min_i:
        return 0
    if w == min_i and min_i != v:
        return 0
    if w == min_i and min_i == v:
        return 1
    if w > max_i:
        den = integrate.quad(O,min_i,max_i)[0]
    else:
        den = integrate.quad(O,min_i,w)[0]
        
    num = O(v)    
    #try:
    prob = num/den
    #except:
    #    if v == w:
    #        prob = 1
    #    else:
    #        prob = 0
            
    return prob
    
def integrand(v,w):
    return v*P(v,w)
    
def gdp_factor(wealth):
    def integrand(v,w):
        return v*P(v,w)
    if wealth == min_i:
        return integrand(wealth,wealth)
    else:
        return integrate.quad(integrand,min_i,wealth,args=(wealth))[0]
    

def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
    return g

def random_distribution(num_balls,num_urns):
    dummy_num_balls = num_balls
    distribution = [0 for urn in range(num_urns)]
    while dummy_num_balls > 0:
        for urn_idx in range(num_urns):
            urn_balls = random.choice(range(dummy_num_balls+1))
            distribution[urn_idx] += urn_balls
            dummy_num_balls = dummy_num_balls - urn_balls
            
    return sorted(distribution, reverse=True)


def random_distribution2(num_balls,num_urns):
    distribution = [0 for urn in range(num_urns)]
    for ball in range(num_balls):
        urn_idx = random.choice(range(len(distribution)))
        distribution[urn_idx] += 1
    return distribution


def all_permutations(num_balls,num_urns):

    rng = list(range(num_balls + 1)) * num_urns
    dists = set(i for i in itertools.permutations(rng, num_urns) if sum(i) == num_balls)
    ds = []
    for dist in dists:
        dist = sorted(list(dist),reverse=True)
        if dist not in ds:
            ds.append(dist)
    return ds


#num_distributions = 20
#distributions = [random_distribution(M,N) for i in range(num_distributions)]
#distributions2 = [random_distribution2(M,N) for i in range(num_distributions)]
equality_dist = [int(M/N) for i in range(N)]
inequality_dist = [M if idx==0 else 0 for idx in range(N)]
#distributions.extend(distributions2)
#distributions.append(equality_dist)
#distributions.append(inequality_dist)

distributions = all_permutations(M,N)

gdps = []
ginis = []
for dist in distributions:
    dist_gdp = 0
    dist_gini = gini(dist)
    for wealth in range(M+1):
        num_with_wealth = dist.count(wealth)
        gdp_wealth_factor = gdp_factor(wealth)
        wealth_gdp = num_with_wealth*gdp_wealth_factor
        dist_gdp += wealth_gdp
       
    gdps.append(dist_gdp)
    ginis.append(dist_gini)
    
    if dist == equality_dist:
        print('EQUALITY GDP:',dist_gdp)
    if dist == inequality_dist:
        print('INEQUALITY GDP:',dist_gdp)
    

best_dist = [[distributions[idx],gdps[idx]] for idx in range(len(distributions)) if gdps[idx] == max(gdps)]
worst_dist = [[distributions[idx],gdps[idx]] for idx in range(len(distributions)) if gdps[idx] == min(gdps)]
print(best_dist)
print(worst_dist)

plt.plot(ginis,gdps,'.')
plt.xlabel('Gini Index')
plt.ylabel('Expected GDP')
plt.legend(['min_i='+str(min_i)+ ', max_i='+str(max_i)])
#plt.savefig(opp_type+'_n4m20_min'+str(min_i)+'max'+str(max_i)+'.png')

plt.figure()
x = [.1*x for x in range(10*M)]
y = [O(x) for x in x]
z = [P(x,10)/10 for x in x]
u = [gdp_factor(x) for x in x]
plt.plot(x,y)
plt.ylabel('Number of Investment Opportunities')
plt.xlabel('Wealth')
plt.legend(['min_i='+str(min_i)+ ', max_i='+str(max_i)])
#plt.savefig(opp_type+'_opp_min'+str(min_i)+'max'+str(max_i)+'.png')
plt.figure()
plt.plot(x,u)
plt.xlabel('Wealth')
plt.ylabel('Expected Individual GDP')
plt.legend(['min_i='+str(min_i)+ ', max_i='+str(max_i)])
#plt.savefig(opp_type+'_iGDP_min'+str(min_i)+'max'+str(max_i)+'.png')
