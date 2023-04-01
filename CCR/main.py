# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import time
import config
import ast
from itertools import permutations
from itertools import combinations
from itertools import chain
import glob
import random
import os
import linecache
import sys
import signal
import json
import scipy.stats as stats
from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import pickle
import joblib

import generation
from generation import *
from ilp import *
from movilp import *
import scipy.stats as ss
# import glovar
import networkx as nx
from queue import PriorityQueue
from profile import *
from mechanism import *
from preference import *
from computefeatures import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def ccr_search_baseline(profile, k=2):  # too slow for m = 16, |committee| = 8
    start = time.perf_counter()
    n = profile.numVoters
    m = profile.numCands
    cand = profile.candMap
    cand = set(cand.keys())
    # print(cand)
    # order_vec = profile.getOrderVectors()
    RankMaps = profile.getRankMaps()
    # print(RankMaps)
    prefcounts = profile.getPreferenceCounts()
    b = len(prefcounts)

    committee0 = set()
    score0 = 0
    root = Node(value=(score0, committee0))
    stackNode = []
    stackNode.append(root)
    # print(stackNode[0].value)
    hashtable = set()
    ed_time = []
    # Jaccard = 0


    # number of nodes
    num_nodes = 1

    L = 0
    best_committee = set()

    while stackNode:
        # print("len=", len(stackNode))
        # Pop new node to explore
        node = stackNode.pop()
        # print("node", node.value)
        (score, committee) = node.value
        # print("info:", score , committee)
        for c in cand - committee:
            new_committee = committee.copy()
            new_committee.add(c)
            tpc = tuple(sorted(new_committee))
            if tpc in hashtable:
                # cache_hits += 1
                # HASHEND = time.perf_counter()
                # HASHTIME += HASHEND - HASHSTART
                continue
            else:
                hashtable.add(tpc)
                new_score = 0
                for x in new_committee:
                    for i in range(b):
                        if RankMaps[i][x] == min(RankMaps[i][x_] for x_ in new_committee):
                            new_score += prefcounts[i] * (m - RankMaps[i][x])

                if len(new_committee) == k:
                    if new_score > L:
                        L = new_score
                        best_committee = new_committee
                        end = time.perf_counter()

                        # temp = len(best_committee.intersection(ilpcommittee)) / len(best_committee.union(ilpcommittee))
                        # if new_score > quality:
                        #     Jaccard = temp
                        ed_time.append([end - start, L])


                    continue
                else:
                    if new_score > L:
                        # print("L, new_score=", L, new_score)
                        L = new_score
                    child_node = Node(value=(new_score, new_committee))
                    stackNode.append(child_node)
                    num_nodes += 1
    # print("DFS final score=", L)
    ed = end - start
    ed_time = [[elem[0], elem[1]/L] for elem in ed_time]
    return best_committee, num_nodes, ed, ed_time


def ccr_search(profile, ml_model, k=2):
    start = time.perf_counter()
    n = profile.numVoters
    m = profile.numCands
    cand = profile.candMap
    cand = set(cand.keys())
    # print(cand)
    # order_vec = profile.getOrderVectors()
    RankMaps = profile.getRankMaps()
    # print(RankMaps)
    prefcounts = profile.getPreferenceCounts()
    b = len(prefcounts)

    committee0 = set()
    score0 = 0
    # root = Node(value=(score0, committee0))
    # stackNode = []
    # stackNode.append(root)
    # print(stackNode[0].value)

    posmat_vec = profile2posmat(profile)
    wmg = profile.getWmg()
    wmg_vec = vectorize_wmg(profile)
    nonnormal_wmg_vec = vectorize_wmg(profile, normal=False)  # 20210821
    pluralitydict = MechanismPlurality().getCandScoresMap(profile, normalize=True)
    pluralityscores = list(pluralitydict.values())
    bordadict = MechanismBorda().getCandScoresMap(profile, normalize=True)
    bordascores = list(bordadict.values())
    copelandscores = copeland_score(wmg)
    maximinscores = maximin_score(wmg)
    X = [list(posmat_vec) + list(pluralityscores) + list(bordascores) + list(wmg_vec) + list(copelandscores) + list(maximinscores)]
    X = np.array(X)
    # print("posmat_vec=", posmat_vec)
    # X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    prediction = ml_model.predict(X)  # 20220224
    # print("prediction=", prediction)


    Q = PriorityQueue()

    # Initialization
    root = (-score0, Node(value=(score0, committee0)))
    Q._put(root)
    hashtable = set()
    ed_time = []
    # Jaccard = 0


    # number of nodes
    num_nodes = 1

    L = 0
    best_committee = set()

    while Q._qsize() != 0:
        # print("len=", len(stackNode))
        # Pop new node to explore
        # node = stackNode.pop()
        pscore, node = Q._get()
        # print("node", node.value)
        (score, committee) = node.value
        # print("info:", pscore, score , committee)
        for c in cand - committee:
            new_committee = committee.copy()
            new_committee.add(c)
            tpc = tuple(sorted(new_committee))
            if tpc in hashtable:
                # cache_hits += 1
                # HASHEND = time.perf_counter()
                # HASHTIME += HASHEND - HASHSTART
                continue
            else:
                hashtable.add(tpc)
                new_score = 0
                for x in new_committee:
                    for i in range(b):
                        if RankMaps[i][x] == min(RankMaps[i][x_] for x_ in new_committee):
                            new_score += prefcounts[i] * (m - RankMaps[i][x])

                if len(new_committee) == k:
                    if new_score > L:
                        L = new_score
                        best_committee = new_committee
                        end = time.perf_counter()

                        # temp = len(best_committee.intersection(ilpcommittee)) / len(best_committee.union(ilpcommittee))
                        # if temp > Jaccard:
                        #     Jaccard = temp
                        ed_time.append([end - start, L])
                    continue
                else:
                    if new_score > L:
                        # print("L, new_score=", L, new_score)
                        L = new_score


                    child_node = Node(value=(new_score, new_committee))
                    # print(prediction, new_committee)
                    # priority_score = -sum([prediction[0][c - 1] for c in new_committee])  # ml before 20220307
                    priority_score = -sum([prediction[0][c - 1]*new_score for c in new_committee])  # ml2 20220307

                    Q._put((priority_score, child_node))
                    num_nodes += 1
    # print("DFS final score=", L)
    ed = end - start
    ed_time = [[elem[0], elem[1] / L] for elem in ed_time]
    return best_committee, num_nodes, ed, ed_time


def ed_curve(ed_time):
    arr = np.array(ed_time)
    # print("ok3", list(arr))
    # print(max(arr[:, 0]))
    max_time = 1.2 * max(arr[:, 0])
    tick = 0.001
    X = np.arange(0.0, max_time, tick)
    Y = np.zeros(len(X))
    [time, U] = ed_time.pop(0)
    for i in range(len(X)):
        if X[i] < time:
            Y[i] = U
        elif len(ed_time) > 0:
            [time, U] = ed_time.pop(0)
            Y[i] = U
        else:
            Y[i] = U
    return X, Y

def read_n_compute_avg_early_discovery(results, tick=0.001, max_time=None, logtime=False):
    mov_times = list()
    max_time2 = 0
    N = 0
    with open(results, 'r', encoding=u'utf-8', errors='ignore') as inf:
        temp = inf.readline()
        entry = temp.strip().split('\t')
        if 'ed_time' not in entry:
            temp = inf.readline()
            entry = temp.strip().split('\t')
        pos = entry.index('ed_time')
        temp = inf.readline()
        while temp:
        # while N<48:
            entry = temp.strip().split('\t')
            if True or len(entry) >= pos + 1:
                mov_time = eval(entry[pos])
                arr = np.array(mov_time)
                discover_time = max(arr[:, 0])
                max_time2 = max(discover_time, max_time2)
                mov_times.append(mov_time)
                N += 1
                # print("{}\t{}".format(entry[0], discover_time))
                print("{}".format(discover_time))
            else:
                print("{}".format(entry[0]))
            temp = inf.readline()
    if max_time is None:
        max_time = max_time2

    if logtime is False:
        X = np.arange(0.0, max_time, tick)
    else:
        max_ind = math.ceil(math.log10(max_time))
        ind = np.arange(-4, max_ind, tick)
        X = 10 ** ind

    Y = np.zeros(len(X))
    n = 0
    for mov_time in mov_times:
        n += 1
        # print("complete {} files.".format(n))
        temp = mov_time.copy()
        [time, U] = temp.pop(0)
        for i in range(len(X)):
            if X[i] < time:
                Y[i] += U
            elif len(temp) > 0:
                [time, U] = temp.pop(0)
                Y[i] += U
            else:
                Y[i] += U
    Y /= N

    return X, Y


class Node:
    def __init__(self, value=None):
        self.value = value

    def __lt__(self, other):
        return 0

    def getvalue(self):
        return self.value


def avg_early_discovery():
    os.chdir(config.results_folder)

    # # generate pickle file and plot
    # X, Y = read_n_compute_avg_early_discovery(config.test_results, tick=0.001, max_time=None, logtime=True)
    # with open(config.curvedump, 'wb') as fo:
    #     pickle.dump([X, Y], fo)
    # plt.plot(X, Y, '-')
    # plt.xscale('log')
    # plt.show()

    # plot single figure
    # with open(config.curvedump,'rb') as f:
    #     [X, Y] = pickle.load(f)
    #
    # # start, end = 1200, 1800
    # start, end = None, None
    # plt.plot(X[start:end], Y[start:end], '-')
    # # plt.xscale('log')
    # plt.show()

    # plot multiple figures
    fig, ax = plt.subplots()
    filenames = glob.glob(config.curvedumps)
    filenames = sorted(filenames)

    filenames = ['results-MBP-baseline-M16N10k-k8-20220308_1k.pickle', 'results-MBP-ML-M16N10k-k8-20220308_1k.pickle',
                 'results-MBP-ML2-M16N10k-k8-20220308_1k.pickle']
    nfiles = len(filenames)
    X = dict()
    Y = dict()
    for i in range(nfiles):
        with open(filenames[i], 'rb') as f:
            print(filenames[i])
            [X[i], Y[i]] = pickle.load(f)

    start, end = 1000, 3000
    # start, end = 6500, 7500
    # start, end = None, None
    colors = ['r', 'g', 'c', 'm', 'b', 'y', 'k']
    # for i in range(nfiles):
        # ax.plot(X[i][start:end], Y[i][start:end], '-', color=colors[i], label=filenames[i].strip().split('_')[0])
    ax.plot(X[0][start:end], Y[0][start:end], '-', color=colors[0], label='baseline')
    ax.plot(X[1][start:end], Y[1][start:end], '-', color=colors[1], label='ml')
    ax.plot(X[2][start:end], Y[2][start:end], '-', color=colors[2], label='ml2')
    # ax.plot(X[3][start:end], Y[3][start:end], '-', color=colors[3], label='DFS(LPML)')

    # plt.xscale('log')
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.legend(loc='lower right')
    plt.xlabel("time /s")
    plt.ylabel("Quality (score/best score)")
    plt.show()


def main():
    global cands
    os.chdir(config.data_folder)
    filenames = glob.glob(config.data_filename)
    filenames = sorted(filenames)
    filenames = filenames[200:1000]

    model_name = config.models_path + 'model-logistic-M16N10k-soc3-100k-experiment-20220223-test_0-9999.pkl'
    mdl = joblib.load(model_name)
    print("{} loaded successfully.".format(model_name))

    random.seed(5)
    print('filename\tsearch result\tsearch time\tILP result\tILP time\tm\tsize\t#nodes\t100ed\ted_time')

    for inputfile in filenames:
        # print(inputfile)
        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
        inf.close()
        # print(cmap, rmapscounts, nvoters)

        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)

        # Currently, we expect the profile to contain complete ordering over candidates. Ties are allowed however.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "soi" and elecType != "csv":
            print("ERROR: unsupported election type")
            exit()

        # order_vec = profile.getOrderVectors()
        # RankMaps = profile.getRankMaps()
        m = profile.numCands
        size = 8
        # print(RankMaps)
        time0 = time.perf_counter()
        c = CCR_ILP(profile, k=size)
        time1 = time.perf_counter()
        # c2, num_nodes, ed, ed_time = ccr_search_baseline(profile, k=size)
        c2, num_nodes, ed, ed_time = ccr_search(profile, mdl, k=size)
        time2 = time.perf_counter()
        # print("committee=", c)
        # print('filename\tsearch result\tsearch time\tILP result\tILP time\tm\tsize\t#nodes')
        # print("time=", time1-time0, time2-time1)
        print('{}\t{}\t{:.4f}\t{}\t{:.4f}\t{}\t{}\t{}\t{:.4f}\t{}'.format(inputfile, c2, time2-time1, c, time1-time0, m, size, num_nodes, ed, ed_time))


def test():
    global cands
    os.chdir(config.data_folder)
    filenames = glob.glob(config.data_filename)
    filenames = sorted(filenames)
    # filenames = filenames[100:101]

    # filenames=['ccr3.soc']

    for inputfile in filenames:
        # print(inputfile)
        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
        inf.close()
        # print(cmap, rmapscounts, nvoters)

        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)

        # Currently, we expect the profile to contain complete ordering over candidates. Ties are allowed however.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "soi" and elecType != "csv":
            print("ERROR: unsupported election type")
            exit()

        # order_vec = profile.getOrderVectors()
        # RankMaps = profile.getRankMaps()
        m = profile.numCands
        n = profile.numVoters
        size = m // 2
        # print(RankMaps)
        # time0 = time.perf_counter()
        # c, model = CCR_ILP4MOV(profile, k=size)
        # time1 = time.perf_counter()
        # x = model._x
        # # print(profile.getRankMaps())
        # print("committee=", c, time1 - time0)
        #
        # time0 = time.perf_counter()
        # mov, model2, new_comm = CCR_MOVILP(profile, x, K=size)
        # time1 = time.perf_counter()
        # print(inputfile, "mov=", mov, time1 - time0)
        # x2 = model2._x
        # print("new committee=", new_comm)
        # r = model2._r
        # for j in range(n):
        #     for i in range(m):
        #         print(r[i, j].X, end = ' ')
        #     print(" ")

        # c2, num_nodes, ed, ed_time = ccr_search_baseline(profile, k=size)
        # print(c2)
        txt = strict_order.generation_one()
        print(txt)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    # main()

    # avg_early_discovery()

    test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
