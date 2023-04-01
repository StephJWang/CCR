"""
    File: 	movilp.py
    Author:	Jun Wang (wangj38@rpi.edu)
    Date:	Mar 22, 2022

    * Copyright (c) 2022, Jun Wang and RPI
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
from gurobipy import GRB, max_, min_, abs_
# import glovar
# import cplex
# from cplex.exceptions import CplexError, CplexSolverError

def CCR_MOVILP(profile, winner_vec, K=2):
    """
    Returns the MOV of given profile under Chamberlain-Courant rule
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
    M = 10 * max(m, n)
    # order_vec = profile.getOrderVectors()
    C0 = set()  # 0-indexed
    for i in range(m):
        if winner_vec[i].X == 1:
            C0.add(i)  # 0-indexed

    RankMaps = profile.getRankMaps()
    # print(order_vec)
    prefcounts = profile.getPreferenceCounts()
    b = len(prefcounts)
    extended_r = []
    for i in range(b):
        extended_r += [RankMaps[i]] * prefcounts[i]

    model = gp.Model('linear_program4ilpmovccr')
    # we set the Gurobi OutputFlag parameter to 0 in order to shut off Gurobi output.
    model.setParam('OutputFlag', 0)

    x = model.addVars(m, vtype=GRB.BINARY, name="x_")  # the new winner committee vector
    y = model.addVars(m, n, vtype=GRB.BINARY, name="y_")  # the new y varabile (y[i,j] whether cand i is represented by voter j)
    y1 = model.addVars(m, n, vtype=GRB.BINARY, name="y1_")  # the y varabile under original winner_vec
    r = model.addVars(m, n, lb=1, ub=m, vtype=GRB.INTEGER, name="r_")  # the new rankmap
    diff1 = model.addVars(m, n, lb=1-m, ub=m-1, vtype=GRB.INTEGER, name="diff1_")  # extended_r[j][i+1] - r[i, j]
    abs1 = model.addVars(m, n, lb=0, ub=m - 1, vtype=GRB.INTEGER, name="abs1_")  # absolute value of diff1
    delta = model.addVars(m, n, vtype=GRB.BINARY, name="delta_")  # delta[i,j] is whether voter j changes cand i's ranking
    delta2 = model.addVars(n, vtype=GRB.BINARY, name="delta2_")  # delta[j] is whether voter j changes her vote
    z = model.addVars(m, n, lb=0, ub=m, vtype=GRB.INTEGER, name="z_")  # r[i,j]*y[i,j]
    z1 = model.addVars(m, n, lb=0, ub=m, vtype=GRB.INTEGER, name="z1_")  # r[i,j]*y1[i,j]
    min_r = model.addVars(n, lb=1, ub=m, vtype=GRB.INTEGER, name="min_r_")

    pairs = list(combinations(range(m), 2))  # 0-indexed
    diff2 = dict()  # r[i1, j] - r[i2, j]
    abs2 = dict()  # absolute value of diff2
    for (i1, i2) in pairs:
        diff2[(i1, i2)] = model.addVars(n, lb=1-m, ub=m-1, vtype=GRB.INTEGER, name='diff2_{}'.format((i1, i2)))
        abs2[(i1, i2)] = model.addVars(n, lb=0, ub=m - 1, vtype=GRB.INTEGER, name='abs2_{}'.format((i1, i2)))

    model.addConstr(gp.quicksum(x[i] for i in range(m)) <= K)
    model.addConstrs(x[i] >= y[i, j] for i in range(m) for j in range(n))
    model.addConstrs(gp.quicksum(y[i, j] for i in range(m)) == 1 for j in range(n))

    model.addConstrs(diff1[i, j] == extended_r[j][i+1] - r[i, j] for i in range(m) for j in range(n))
    model.addConstrs(abs1[i, j] == abs_(diff1[i, j]) for i in range(m) for j in range(n))
    model.addConstrs(abs1[i, j] >= delta[i, j] for i in range(m) for j in range(n))
    model.addConstrs(abs1[i, j] <= M*delta[i, j] for i in range(m) for j in range(n))
    model.addConstrs(gp.quicksum(delta[i, j] for i in range(m)) >= delta2[j] for j in range(n))
    model.addConstrs(delta2[j] >= delta[i, j] for i in range(m) for j in range(n))

    model.addConstrs(gp.quicksum(r[i, j] for i in range(m)) == m*(m + 1)//2 for j in range(n))
    model.addConstrs(diff2[(i1, i2)][j] == r[i1, j] - r[i2, j] for (i1, i2) in pairs for j in range(n))
    model.addConstrs(abs2[(i1, i2)][j] == abs_(diff2[(i1, i2)][j])  for (i1, i2) in pairs for j in range(n))
    model.addConstrs(abs2[(i1, i2)][j] >= 1 for (i1, i2) in pairs for j in range(n))

    model.addConstrs(min_r[j] == min_([r[ii, j] for ii in C0]) for j in range(n))
    for i in range(m):
        if winner_vec[i].X == 0:
            model.addConstrs(y1[i, j] == 0 for j in range(n))
        else:
            model.addConstrs((y1[i, j] == 0) >> (r[i, j] >= min_r[j] + 1) for j in range(n))
            model.addConstrs((y1[i, j] == 1) >> (r[i, j] == min_r[j]) for j in range(n))

    # the score under original winner committee
    alpha1 = gp.quicksum(m * y1[i, j] - z1[i, j] for j in range(n) for i in range(m))
    # the score under new winner committee
    alpha = gp.quicksum(m * y[i, j] - z[i, j] for j in range(n) for i in range(m))
    model.addConstr(alpha >= alpha1 + 1)

    model.addConstrs(z[i, j] <= m * y[i, j] for i in range(m) for j in range(n))
    model.addConstrs(z[i, j] <= r[i, j] for i in range(m) for j in range(n))
    model.addConstrs(z[i, j] >= r[i, j] - (1 - y[i, j])*m for i in range(m) for j in range(n))

    model.addConstrs(z1[i, j] <= m * y1[i, j] for i in range(m) for j in range(n))
    model.addConstrs(z1[i, j] <= r[i, j] for i in range(m) for j in range(n))
    model.addConstrs(z1[i, j] >= r[i, j] - (1 - y1[i, j]) * m for i in range(m) for j in range(n))

    votes = gp.quicksum(delta2[j] for j in range(n))
    model.setObjective(votes, gp.GRB.MINIMIZE)
    model.optimize()
    # time3 = time.perf_counter()
    # print("Gurobi solve time = ", time3 - time2)
    model._x = x
    model._y = y
    model._r = r
    committee = set()

    if model.Status == gp.GRB.Status.OPTIMAL:

        d = math.ceil(model.objVal)
        for i in range(m):
            temp = model.getVarByName("x_[{}]".format(i))
            # print(temp)
            if temp.x == 1:
                committee.add(i+1)

    else:
        print("No solution!")
        d = 2 * n
    return d, model, committee
