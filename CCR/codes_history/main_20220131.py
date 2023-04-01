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
from generation import *
from ilp import *
import scipy.stats as ss
import glovar
import networkx as nx

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def ccr_search_baseline(profile, k=2):  # too slow for m = 16, |committee| = 8
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
                    continue
                else:
                    if new_score > L:
                        # print("L, new_score=", L, new_score)
                        L = new_score
                    child_node = Node(value=(new_score, new_committee))
                    stackNode.append(child_node)
                    num_nodes += 1
    # print("DFS final score=", L)
    return best_committee, num_nodes


class Node:
    def __init__(self, value=None):
        self.value = value

    def __lt__(self, other):
        return 0

    def getvalue(self):
        return self.value


def main():
    global cands
    os.chdir(config.data_folder)
    filenames = glob.glob(config.data_filename)
    filenames = sorted(filenames)
    filenames = filenames[0:1000]
    random.seed(5)
    print('filename\tsearch result\tsearch time\tILP result\tILP time\tm\tsize\t#nodes')

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
        c2, num_nodes = ccr_search_baseline(profile, k=size)
        time2 = time.perf_counter()
        # print("committee=", c, c2)
        # print('filename\tsearch result\tsearch time\tILP result\tILP time\tm\tsize\t#nodes')
        # print("time=", time1-time0, time2-time1)
        print('{}\t{}\t{:.4f}\t{}\t{:.4f}\t{}\t{}\t{}'.format(inputfile, c2, time2-time1, c, time1-time0, m, size, num_nodes))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
