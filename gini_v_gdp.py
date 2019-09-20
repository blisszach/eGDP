# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:30:20 2019

@author: Zachery.Bliss
"""

import random
import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import itertools


def O(c):
    if c < min_c or c > max_c:
        return 0
    else:
        if opp_type == 'uniform':
            return 1
        if opp_type == 'sin':
            return math.sin((c-min_c)*math.pi/(max_c-min_c))
        if opp_type == 'exp':
            return math.exp(c-min_c)
        if opp_type == 'negexp':
            return math.exp(-(c-min_c))


def eGDPvGini(N,M,min_c,max_c,opp_type):
    
    def all_permutations(num_balls,num_urns):
        rng = list(range(num_balls + 1)) * num_urns
        dists = set(i for i in itertools.permutations(rng, num_urns) if sum(i) == num_balls)
        ds = []
        for dist in dists:
            dist = sorted(list(dist),reverse=True)
            if dist not in ds:
                ds.append(dist)
        return ds
    
    def gini(x):
        mad = np.abs(np.subtract.outer(x, x)).mean()
        rmad = mad/np.mean(x)
        g = 0.5 * rmad
        return g

    def gdp_factor(wealth):
        
        def integrand(c,w):
            return c*P(c,w)
        if wealth == min_c:
            return integrand(wealth,wealth)
        else:
            return integrate.quad(integrand,min_c,wealth,args=(wealth))[0]

    def P(c,w):
        if c > max_c or c < min_c or c > w:
            return 0
        if w == min_c == c:
            return 1
        num = O(c)
        den = integrate.quad(O,min_c,min(max_c,w))[0] 
        prob = num/den
          
        return prob    
    

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
        

    return [ginis,gdps]


def getPlots(ginis,gdps,opp_type):
    
    plt.figure()
    plt.plot(ginis,gdps,'.')
    plt.xlabel('Gini Index')
    plt.ylabel('Expected GDP')
    plt.legend(['min_c='+str(min_c)+ ', max_c='+str(max_c)])
    plt.savefig(opp_type+'_n4m20_min'+str(min_c)+'max'+str(max_c)+'.png')

    plt.figure()
    slices = 10
    x = [x/slices for x in range(slices*M)]
    y = [O(x) for x in x]

    plt.plot(x,y)
    plt.ylabel('Number of Investment Opportunities')
    plt.xlabel('Wealth')
    plt.legend(['min_c='+str(min_c)+ ', max_c='+str(max_c)])
    plt.savefig(opp_type+'_opp_min'+str(min_c)+'max'+str(max_c)+'.png')

N = 4
M = 20

min_c = 0
max_c = M

opp_types = ['uniform','sin','exp','negexp']
mins_maxs = [[0,M],[5,M],[6,M],[0,15],[6,15]]

for opp_type in opp_types:
    for min_max in mins_maxs:
        min_c = min_max[0]
        max_c = min_max[1]
        xx = eGDPvGini(N,M,min_c,max_c,opp_type)
        getPlots(xx[0],xx[1],opp_type)

