from itertools import permutations
from itertools import combinations
# from math import comb


def isConsistent(rankmap, edge):
    return rankmap[edge[0]] < rankmap[edge[1]]


m = 16  # number of candidates
n = 10000
# num_edges = comb(m, 2)  # number of undirected edges
num_edges = m*(m-1)//2
tie_breaking_order = list(permutations(list(range(1, 1 + m)), 2))
I = list(range(1, m + 1))
cands = {x: 'c' + str(x) for x in I}
R = list(permutations(cands, m))  # All possible ballot signatures (soc), R[0], R[1], ...
edges = set(combinations(list(range(1, 1 + m)), 2))  # Undirected edges
all_arcs = set(permutations(list(range(1, m + 1)), 2))  # all possible directed edges

c = dict()
for (i, j) in edges:
    if (i, j) not in c:
        c[(i, j)] = dict()
        c[(j, i)] = dict()
    for k in range(len(R)):
        s = R[k]
        # print(s)
        rankmap = {s[i]: i + 1 for i in range(len(s))}
        c[(i, j)][k] = 1 if isConsistent(rankmap, (i, j)) is True else -1
        c[(j, i)][k] = -c[(i, j)][k]