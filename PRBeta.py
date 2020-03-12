import numpy as np

def depth_search(res_Graph, V, s, visited):
    # res_Graph = residual graph
    # s = source node
    # V = number of vertices
    # visited = set of visited nodes, v[i]=True if node i was visited
    stack = [s]
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            stack.extend([u for u in range(V) if res_Graph[v][u] > 0])

def push(F, excess, u, v):
    send = min(excess[u], C[u][v] - F[u][v])
    F[u][v] += send
    F[v][u] -= send
    excess[u] -= send
    excess[v] += send
    return F, excess

def relabel(F, C, height, u):
    # if possible, find smallest new height making a push possible,
    n = len(C)
    min_h = float('inf')
    for v in range(n):
        if C[u][v] - F[u][v] > 0:
            min_h = min(min_h, height[v])
            height[u] = min_h + 1
    return height, min_h

def discharge(F, C, excess, seen, height, u):
    n = len(C)
    while excess[u] > 0:
        if seen[u] < n: # check next neighbour
            v = seen[u]
            if C[u][v] - F[u][v] > 0 and height[u] > height[v]:
                push(F, excess, u, v)
            else:
                seen[u] += 1
        else:
            relabel(F, C, height, u)
            seen[u] = 0
    return seen

def MaxFlow(C, s, t):
     # C = capacity matrix
     # t = destination node
     n = len(C)
     F = [[0] * n for i in range(n)]

     # residual capacity  u -> v = C[u][v] - F[u][v]
     height = [0] * n # height of node
     excess = [0] * n # flow into node minus flow from node
     seen   = [0] * n # neighbours seen since last relabel
    
     nodelist = [i for i in range(n) if i != s and i != t]
 
     height[s] = n   # longest path s -> t is < n
     excess[s] = float("inf")
     for v in range(n):
         push(F, excess, s, v)
 
     p = 0
     while p < len(nodelist):
         u = nodelist[p]
         old_height = height[u]
         discharge(F, C, excess, seen, height, u)
         if height[u] > old_height:
             nodelist.insert(0, nodelist.pop(p)) # move to top
             p = 0 # start from top
         else:
             p += 1
     #return sum(F[s])
     print ('done push relabel')
     V = n
     visited = np.zeros(V, dtype=bool)
     depth_search(np.subtract(C,F), V, s, visited)
     print ('finding cuts')
     cuts = []

     for u in range(V):
        for v in range(V):
            #(u,v) is a cut if we have visited u but not v and (u,v) exists
            if visited[u] and not visited[v] and C[u][v]:
                cuts.append((u, v))
     return cuts
