"""
    File: 	features.py
    Author:	Jun Wang (wangj38@rpi.edu)
    Date:	Feb 20, 2022

    * Copyright (c) 2022, Jun Wang and RPI
    * All rights reserved.
    *
    * Developed by: Jun Wang
"""
import prefpy_io
import math
import config
from profile import Profile
import json
import pickle
import logging
import numpy as np
from numpy import linalg as LA
import networkx as nx
from networkx.utils import py_random_state
from networkx.generators.classic import complete_graph
import os
import itertools
from profile import *
from mechanism import *
from preference import *
from ilp import *
from generation import *
from scipy.special import comb
import random
import joblib
from computefeatures import *
from main import *
import matplotlib.pyplot as plt
import scipy.stats as ss
import signal
# import glovar


# write the features and ground truth to files, plain text csv and pickled
# input is the generated features and extracted ground truth
# ouputs nothing
def write_features(xrows, yrows, features, filenames):
    outputprefix = config.featuresoutputprefix
    with open(outputprefix+'.README','w') as fo:
        fo.write(','.join(features)+';'+'isdist')
    data = {'X': xrows, 'y': yrows, 'features': features, 'filenames': filenames}
    with open(outputprefix+'.json','w') as fo:
        fo.write(json.dumps(data))


def read_dist_result(inputfile):
    Y = dict()
    temp = inputfile.readline()
    # temp = inputfile.readline()
    # print(temp)
    while temp:
        information = temp.strip().split("\t")
        if information[0] == 'filename':
            temp = inputfile.readline()
            continue
        # print(information[1])
        # order = tuple(map(int, information[1].strip('()').split(', ')))
        order = eval(information[1])
        # print(order)
        if information[0] in Y:
            Y[information[0]].append([order, float(information[2])])
        else:
            Y[information[0]] = [[order, float(information[2])]]
        temp = inputfile.readline()
    return Y


def compute_avg_dist(inputfiles):
    mean = 0
    N = 0
    for inputfile in inputfiles:
        inf = open(inputfile, 'r')
        temp = inf.readline()
        while temp:
            information = temp.strip().split("\t")
            mean += float(information[2])
            N += 1
            temp = inf.readline()
        print("{} completed.".format(inputfile))
        inf.close()
    print("N = ", N)
    mean /= N
    print("Average dist = ", mean)
    return mean


@py_random_state(2)
def my_random_dag(n, m, seed=None, directed=False):
    """Returns a $G_{n,m}$ random graph.

    In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set
    of all graphs with $n$ nodes and $m$ edges.

    This algorithm should be faster than :func:`dense_gnm_random_graph` for
    sparse graphs.

    Parameters
    ----------
    n : int
        The number of nodes.
    m : int
        The number of edges.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True return a directed graph

    See also
    --------
    dense_gnm_random_graph

    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))

    if n == 1:
        return G
    max_edges = n * (n - 1)
    if not directed:
        max_edges /= 2.0
    if m >= max_edges:
        return complete_graph(n, create_using=G)

    nlist = list(G)
    edge_count = 0
    while edge_count < m:
        # generate random edge,u,v
        u = seed.choice(nlist)
        v = seed.choice(nlist)
        if u == v or G.has_edge(u, v) or nx.has_path(G, v, u):
            continue
        else:
            G.add_edge(u, v)
            edge_count = edge_count + 1
    return G


# main function, generate features and extract ground truth
# output xrows for features and corresponding yrows for winners
# randomly generate data points of profiles and m(m-1)/2-length sequences
def generation_extraction():
    def handler(signum, frame):
        raise AssertionError
    features = list()
    xrows = list()
    yrows = list()
    filenames = list()
    # seqs = list()
    # ---------------------------------------read other small data as features---------------------------------------
    os.chdir(config.dist_folder)
    # result = open(config.write_distfile, 'w+')
    iters = 100000
    start_ind = 0
    # m = glovar.m
    # n = glovar.n
    m = 16
    n = 10000
    size = 8
    # num_samples_per_profile = 10
    iter = 0
    print(iters)
    while iter < iters:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(20)

            filename = str(iter).zfill(len(str(iters)))
            filenames.append(filename)
            txt = strict_order(1, m, n, True, 3).generation_one()
            cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_txt(txt)

            profile = Profile(cmap, preferences=[])
            Profile.importPreflibVotes(profile, cmap, rmaps, rmapscounts, nvoters)

            posmat_vec = profile2posmat(profile)
            wmg = profile.getWmg()
            # print(wmg)
            wmg_vec = vectorize_wmg(profile, normal=True)
            # nonnormal_wmg_vec = vectorize_wmg(profile, normal=False)
            # print(wmg_vec)
            # computes the plurality scores of candidates given an input profile
            # input: profile
            # output: m-vector of plurality scores of candidates, normalized by n
            pluralitydict = MechanismPlurality().getCandScoresMap(profile, normalize=True)
            pluralityscores = list(pluralitydict.values())

            # computes the Borda scores of candidates given an input profile
            # input: profile
            # output: m-vector of Borda scores of candidates, normalized by n(m-1)
            bordadict = MechanismBorda().getCandScoresMap(profile, normalize=True)
            bordascores = list(bordadict.values())
            copelandscores = copeland_score(wmg)
            maximinscores = maximin_score(wmg)

            I = range(1, m + 1)
            E = nx.DiGraph()
            E.add_nodes_from(I)
            for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
                if wmg[cand1][cand2] >= 0:  # including w=0 edges
                    E.add_edge(cand1, cand2, weight=wmg[cand1][cand2])


            # print(E.edges)
            c = CCR_ILP(profile, k=size)
            committee_binary = [0]*m
            for candidate in c:
                committee_binary[candidate - 1] = 1


            xrow = list(posmat_vec) + list(pluralityscores) + list(bordascores) + list(wmg_vec) + list(copelandscores) + list(maximinscores)
            # print("X size=", len(xrow))
            features = ['posmat_vec_m^2', 'pluralityscores_m', 'bordascores_m', 'wmg_vec_m_m-1', 'copelandscores', 'maximinscores']
            # └-------------------------------------------------------------┘

            xrows.append(xrow)
            # extract y
            yrow = committee_binary
            # print("yrow=", yrow)
            yrows.append(yrow)
            iter += 1
            # print(maxedge)
            print("{}\t{}".format(iter, yrow))
            # print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(iter, SFD, maxedge, norm1, norm2, norm3, norm4, y_value), file=result)
            signal.alarm(0)
        except AssertionError:
            print("timeout")
        # print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(iter, SFD, kendall, norm1, norm2, norm3, norm4, GED, lp, dist))
    print("size=", len(yrows))
    write_features(xrows, yrows, features, filenames)
    # result.close()


def test_mechanism():
    os.chdir(config.profile_folder)
    filenames = glob.glob(config.profile_filenames)
    filenames = sorted(filenames)
    filenames = filenames[0:1]
    print(filenames)

    for inputfile in filenames:
        inf = open(inputfile, 'r')
        # print(os.getcwd())
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)

        inf.close()
        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)


        print(min(profile.candMap.keys()))

        posmat_vec = profile2posmat(profile)
        # borda = borda_score(values)
        print(posmat_vec)

        bordalist = MechanismPlurality().getCandScoresMap(profile, normalize=True)
        bordalist2 = MechanismPlurality().getCandScoresMap(profile)
        print(bordalist, bordalist2)


def test():
    txt = strict_order(1, 3, 10, False, 1).generation_one()
    print(txt)
    cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_txt(txt)
    print("txt=", txt)

    profile = Profile(cmap, preferences=[])
    Profile.importPreflibVotes(profile, cmap, rmaps, rmapscounts, nvoters)

    posmat_vec = profile2posmat(profile)
    # borda = borda_score(values)
    print(posmat_vec)

    bordalist = MechanismPlurality().getCandScoresMap(profile, normalize=True)
    bordalist2 = MechanismPlurality().getCandScoresMap(profile)
    print(bordalist, bordalist2)


def read_dist(dist_result):
    SFD, maxedge, l1, l2, l_fro, l_inf, y_value = list(), list(), list(), list(), list(), list(), list()
    with open(dist_result, 'r', encoding=u'utf-8', errors='ignore') as inf:
        temp = inf.readline()
        while temp:
            entry = temp.strip().split('\t')
            SFD.append(float(entry[1]))
            maxedge.append(float(entry[2]))
            l1.append(float(entry[3]))
            l2.append(float(entry[4]))
            l_fro.append(float(entry[5]))
            l_inf.append(float(entry[6]))
            y_value.append(float(entry[7]))
            temp = inf.readline()
    return SFD, maxedge, l1, l2, l_fro, l_inf, y_value


def scatter3Dplot():
    os.chdir(config.results_folder)
    filename = config.write_distfile
    SFD, maxedge, l1, l2, l_fro, l_inf, y_value = read_dist(filename)
    print("SFD=", SFD[0:10])
    print("y_value=", y_value[0:10])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(SFD, l1, y_value, marker='o', s=1)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == '__main__':
    # FORMAT = "%(asctime)s %(levelname)s %(module)s %(lineno)d %(funcName)s:: %(message)s"
    # logging.basicConfig(filename='common.log', filemode='a', level=logging.ERROR, format=FORMAT)
    # main()

    generation_extraction()
    # print("Hello world!")

    # scatter3Dplot()
