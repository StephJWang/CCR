"""
    File: 	ilp.py
    Author:	Jun Wang (wangj38@rpi.edu)
    Date:	Jul 14, 2021

    * Copyright (c) 2021, Jun Wang and RPI
    * All rights reserved.
    *
    * Developed by: Jun Wang
"""
import numpy as np
import time
import io
import os
import sys
import prefpy_io
import math
import config
from profile import Profile
from itertools import permutations
from itertools import combinations
from itertools import chain
import gurobipy as gp
from gurobipy import GRB, max_
import networkx as nx
from itertools import permutations
from itertools import combinations
from itertools import chain
import glovar
# import cplex
# from cplex.exceptions import CplexError, CplexSolverError


def CCR_ILP(profile, k=2):
    """
    Returns the minimum number of votes needed to realize given sequence.
    :param profile: the election profile in Preflib format
    :param sequence: the list of edges that will be changed to
    :param package: the linear program solver package, is 'Gurobi' by default, can be
                    also set as 'cplex'
    :param vartype: the type of the variables in LP, is 'I' (integer) by default, can be
                    also set as 'C' (continuous)
    :return: an integer that corresponds to the "distacne" from original order to pi
    """
    # start = time.perf_counter()

    n = profile.numVoters
    m = profile.numCands
    # order_vec = profile.getOrderVectors()
    RankMaps = profile.getRankMaps()
    # print(order_vec)
    prefcounts = profile.getPreferenceCounts()
    b = len(prefcounts)
    print(prefcounts)

    model = gp.Model('linear_program4ilpccr')
    # we set the Gurobi OutputFlag parameter to 0 in order to shut off Gurobi output.
    model.setParam('OutputFlag', 0)

    x = model.addVars(m, vtype=GRB.BINARY, name="x_")
    y = model.addVars(m, b, vtype=GRB.BINARY, name="y_")
    model.addConstr(gp.quicksum(x[i] for i in range(m)) == k)
    model.addConstrs(x[i] >= y[i, j] for i in range(m) for j in range(b))
    # model.addConstrs(y[i, j] - (m - (order_vec[j].index(i + 1) + 1) - max_(x[i]*(m - (order_vec[j].index(i + 1) + 1)))) >= x[i] for i in range(m) for j in range(b))
    #
    # alpha = gp.quicksum(x[i]*(m - (order_vec[j].index(i + 1) + 1)) for i in range(m) for j in range(b))

    model.addConstrs(x[i] - y[i, j] + m - RankMaps[j][i+1] <= max_(x[i] * (m - RankMaps[j][i+1]) for i in range(m))  for i in range(m) for j in range(b))

    alpha = gp.quicksum(y[i, j] * (m - RankMaps[j][i+1]) for i in range(m) for j in range(b))
    model.setObjective(alpha, gp.GRB.MAXIMIZE)
    # time2 = time.perf_counter()
    # print("Gurobi adding time = ", time2 - end)

    model.optimize()
    # time3 = time.perf_counter()
    # print("Gurobi solve time = ", time3 - time2)
    committee = set()

    if model.Status == gp.GRB.Status.OPTIMAL:

        d = math.ceil(model.objVal)
        print("d=", d)
        for i in range(m):
            temp = model.getVarByName("x_{}".format(i))
            if temp == 1:
                committee.add(temp)
    else:
        print("No solution!")
        # d = -1
    return committee#, model


def rank(order_vec, i, j):
    return order_vec[j].index(i + 1) + 1
