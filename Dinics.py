#Dinic Algorithm
import numpy as np
#build level graph by using BFS
def Bfs(C, F, s, t):  # C is the capacity matrix
        n = len(C)
        queue = []
        queue.append(s)
        global level
        level = n * [0]  # initialization
        level[s] = 1  
        while queue:
            k = queue.pop(0)
            for i in range(n):
                    if (F[k][i] < C[k][i]) and (level[i] == 0): # not visited
                            level[i] = level[k] + 1
                            queue.append(i)
        return level[t] > 0

#search augmenting path by using DFS
def Dfs(C, F, k, cp):
        tmp = cp
        if k == len(C)-1:
            return cp
        for i in range(len(C)):
            if (level[i] == level[k] + 1) and (F[k][i] < C[k][i]):
                f = Dfs(C,F,i,min(tmp,C[k][i] - F[k][i]))
                F[k][i] = F[k][i] + f
                F[i][k] = F[i][k] - f
                tmp = tmp - f
        return cp - tmp

def dfs2(rGraph, V, s, visited):
    stack = [s]
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            stack.extend([u for u in range(V) if rGraph[v][u]])

#calculate max flow
#_ = float('inf')
def MaxFlowDinic(C,s,t):
        n = len(C)
        F = [n*[0] for i in range(n)] # F is the flow matrix
        flow = 0
        while(Bfs(C,F,s,t)):
               flow = flow + Dfs(C,F,s,100000)
        #return flow
        V=n
        visited = np.zeros(V, dtype=bool)
        dfs2(np.subtract(C, F), V, s, visited)
        cuts = []

        for i in range(V):
            for j in range(V):
                if visited[i] and not visited[j] and C[i][j]:
                    cuts.append((i, j))
        return cuts


if __name__ == "__main__":

    graph = [[0, 16, 13, 0, 0, 0],
              [0, 0, 10, 12, 0, 0],
              [0, 4, 0, 0, 14, 0],
              [0, 0, 9, 0, 0, 20],
              [0, 0, 0, 7, 0, 4],
              [0, 0, 0, 0, 0, 0]]
#   graph = [[0, 4, 0, 5, 1, 0, 0],
#             [4, 0, 4, 0, 10, 0, 0],
#             [0, 4, 0, 0, 0, 10, 6],
#             [5, 0, 0, 0, 5, 0, 0],
#             [1, 0, 0, 5, 0, 5, 0],
#             [0, 0, 10, 0, 5, 0, 4],
#             [0, 0, 6, 0, 0, 4, 0]]
    s=0
    t= 5
    V = len(graph)
    parent = np.zeros(V, dtype='int32')

    #print (bfs(rGraph, V, s, t, parent))
    print (max_flow(graph, 0, 5))



"""
#-------------------------------------
# make a capacity graph
# node   s   o   p   q   r   t
C = [[ 0, 3, 3, 0, 0, 0 ],  # s
     [ 0, 0, 2, 3, 0, 0 ],  # o
     [ 0, 0, 0, 0, 2, 0 ],  # p
     [ 0, 0, 0, 0, 4, 2 ],  # q
     [ 0, 0, 0, 0, 0, 2 ],  # r
     [ 0, 0, 0, 0, 0, 3 ]]  # t

source = 0  # A
sink = 5    # F
print "Dinic's Algorithm"
max_flow_value = MaxFlow(C, source, sink)
print "max_flow_value is", max_flow_value
"""