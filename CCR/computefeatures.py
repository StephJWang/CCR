"""
    File: 	computefeatures.py
    Author:	Jun Wang (wangj38@rpi.edu)
    Date:	November 10 , 2021

    * Copyright (c) 2021, Jun Wang and RPI
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
from generation import *
from scipy.special import comb
import random
import joblib
import matplotlib.pyplot as plt
import scipy.stats as ss

# Previous profile2posmat might be wrong! --Jun Wang, Nov 12, 2019

# creates a positional matrix (define below) and vectorizes it
# input: profile
# intermediate: positional matrix posmat
#   posmat[i][j] = # voters ranking candidate i in position j
# output: vectorized positional matrix, sorted by candidate, then by position,
#   normalized by no. of voters
def profile2posmat(profile):
    prefcounts = profile.getPreferenceCounts()
    len_prefcounts = len(prefcounts)
    m = profile.numCands
    n = profile.numVoters
    ordering = profile.getOrderVectors()
    posmat = np.zeros((m,m))
    delta = min(profile.candMap.keys())
    for i in range(len_prefcounts):
        for j in range(len(ordering[i])):
            posmat[ordering[i][j] - delta][j] += prefcounts[i]
    posmat_vec = posmat.flatten()
    posmat_vec_normalized = list(1.*np.array(posmat_vec)/n)
    return posmat_vec_normalized


# just vectorizes the wmg
# input: wmg
# output: vectorized weighted majority graph. sorted by candidates, then by opponents,
#   normalized by no. of voters
def vectorize_wmg(profile, normal=True):
    wmg = profile.getWmg(normalize=normal)
    wmg_vec = [wmg[i][j] for i in sorted(wmg.keys()) for j in sorted(wmg[i].keys())]
    return wmg_vec


# computes copeland scores of candidates given an input wmg
# input: wmg
# output: m-vector of copeland scores of candidates, normalized by number of candidates
def copeland_score(wmg):
    m = len(wmg)
    # copelandscores = [(np.sum(np.array(list(wmg[i].values()))>0) - np.sum(np.array(list(wmg[i].values()))<0)) for i in range(m)]
    copelandscores = [(np.sum(np.array(list(wmg[i].values())) > 0) - np.sum(np.array(list(wmg[i].values())) < 0)) for i in wmg.keys()]  # 20220222
    copelandscores_normalized = list(1.*np.array(copelandscores)/m)
    return copelandscores_normalized


# computes maximin scores of candidates given an input wmg
# input: wmg
# output: m-vector of maximin scores of candidates, normalized by number of voters
#   maximin score: for each candidate: lowest margin of victory (negative if defeated)
#                  values may be negative
def maximin_score(wmg):
    m = len(wmg)
    # n = np.sum(np.abs([wmg[0][i] for i in range(1, m)]))
    n = np.sum(np.abs([wmg[1][i] for i in range(2, m+1)]))  # 20220222
    # maximinscores = [np.min(list(wmg[i].values())) for i in range(m)]
    maximinscores = [np.min(list(wmg[i].values())) for i in wmg.keys()]  # 20220222
    maximinscores_normalized = list(1.*np.array(maximinscores)/n)
    return maximinscores_normalized


def reversed(e):
    return (e[1], e[0])


def extended_index(seq, e):
    if e in seq:
        return seq.index(e)
    elif reversed(e) in seq:
        return seq.index(reversed(e))
    else:
        return -1


def extended_KDT(A, B, normalized=True):
    """
    Returns an integer that represents the Kendall tau distance between two
    ranking lists A and B, i.e. the number of pairwise disagreements between
    two ranking lists A and B.
    :param A: ranking list
    :param B: ranking list
    :param normalized: if True, then return normalized Kendall tau distance.
    :return: the Kendall's tau distance
    """
    if len(A) != len(B):
        raise ValueError("All inputs to `kendalltau` must be of the same size, "
                         "found x-size %s and y-size %s" % (len(A), len(B)))
    elif not len(A) or not len(B):
        return np.nan  # Return NaN if arrays are empty

    pairs = combinations(B, 2)
    distance = 0

    for x, y in pairs:
        b = B.index(x) - B.index(y)
        a = extended_index(A, x) - extended_index(A, y)

        # if discordant (different signs)
        if a * b < 0:
            distance += 1

    if normalized is False:
        return distance
    else:
        return distance / comb(len(B), 2, exact=True)


def get_full_sequence(G):
    L = [(d['weight'], (u, v)) for (u, v, d) in G.edges(data=True)]
    L.sort(key=lambda x: (-x[0], x[1]))
    L = [x[1] for x in L]
    return L


def extended_generalized_KDT(A, B, normalized=True):
    """
    Returns an integer that represents the Kendall tau distance between two
    ranking lists A and B, i.e. the number of pairwise disagreements between
    two ranking lists A and B.
    :param A: ranking list
    :param B: ranking list
    :param normalized: if True, then return normalized Kendall tau distance.
    :return: the Kendall's tau distance
    """
    if len(A) == 1:
        return 0
    elif len(A) < len(B):
        B = [x for x in B if x in A or reversed(x) in A]
    elif not len(A) or not len(B):
        return np.nan  # Return NaN if arrays are empty

    pairs = combinations(B, 2)
    distance = 0

    for x, y in pairs:
        b = B.index(x) - B.index(y)
        a = extended_index(A, x) - extended_index(A, y)
        # if discordant (different signs)
        if a * b < 0:
            distance += 1

    if normalized is False:
        return distance
    else:
        # print("A=", A, "B=", B, distance)
        # print("leb=", len(B), "cb2=", comb(len(B), 2, exact=True))
        return distance / comb(len(B), 2, exact=True)


def extended_SFD(A, B, normalized=True, aligned=False):
    """
    Returns an integer that represents the total index difference between two
    ranking lists A and B.
    :param A: ranking list
    :param B: ranking list (original)
    :return: the total index difference
    """
    if not len(A) or not len(B):
        return np.nan  # Return NaN if arrays are empty
    if len(A) < len(B):
        if aligned is False:
            B = [x for x in B if x in A or reversed(x) in A]

    total = 0
    m_ = len(A)
    m = len(B)
    for x in A:
        if x in B:
            total += abs(A.index(x) - B.index(x))
        elif reversed(x) in B:
            total += abs(A.index(x) - 2*m_ + 1 + B.index(reversed(x)))

    if normalized is False:
        return total
    else:
        return total / max_SFD(m, m_)


def max_SFD(m, m_):
    m2 = math.ceil((m + m_)/2)
    max_value = math.floor(m2**2/2) - (math.floor(m2/2) if (m-m_)%2 != 0 else 0)
    return max_value


def wmg2rankmat(G):
    """
    compute ranking matrix of given directed graph
    :param G: a directed graph with weights
    :return: a ranking matrix
    """
    i = 0
    weight = [d['weight'] for u, v, d in G.edges(data=True)]
    ranking = ss.rankdata(weight, method='max')
    E = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        E.add_edge(u, v, weight=ranking[i])
        i += 1
    mat = nx.to_numpy_matrix(E, nodelist=sorted(E.nodes))
    adj = mat-mat.transpose()
    # print("adj=", adj)
    return adj


def seq2rankmat(seq, m):
    """
    compute ranking matrix of given sequence
    :param seq: a list of tuples: sequence
    :return: a ranking matrix
    """
    E = nx.DiGraph()
    E.add_nodes_from(glovar.I) # 20211207 resolve errorL: networkx.exception.NetworkXError: Node 3 in nodelist is not in G
    num_edges=m*(m-1)/2
    i = 0
    for u, v in seq:
        E.add_edge(u, v, weight=num_edges-i)
        i += 1
    # print(E.edges)
    mat = nx.to_numpy_matrix(E, nodelist=glovar.I)
    adj = mat - mat.transpose()
    # print("adj=", adj)
    return adj


def matrix_distance(G, seq, type):
    """
    Compute the matrix distance from original graph G to given sequence
    :param G: DiGraph: directed graph with weights
    :param seq: list of tuples: sequence of adding edges
    :return: float: matrix distance
    """
    mat = wmg2rankmat(G)
    mat2 = seq2rankmat(seq, glovar.m)
    mat1 = np.multiply(mat, mat2 != 0)
    return LA.norm(mat1-mat2, ord=type)
