

graph = [[1,0,1,1,0],
         [0,1,1,0,1],
         [1,1,1,1,0],
         [1,0,1,1,0],
         [0,1,0,0,1]
         ]


visited = []
def dfs(g,node,visited):
    if node == 4:
        print("FOUND it")
        visited.append(node)
        print(visited)
        return visited
    if node not in visited:
        visited.append(node)
        for neighbour in range(len(g[node])):
            if g[node][neighbour] == 1:
                ans = dfs(g, neighbour, visited)
                if ans != None:
                    return ans

dfs(graph,0,visited)


def bfs(g,start,end):
    
    queue = [start]
    visited = []
    while queue != []:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            if node == end:
                return visited
            for neighbour in range(len(g[node])):
                if g[node][neighbour] == 1:
                    queue.append(neighbour)

print(bfs(graph,0,4))


def dijkstra(nodes, distances):

    unvisited = {node: None for node in nodes}

    visited = {}
    current = 'B'

    currentDistance = 0
    unvisited[current] = currentDistance
    while True:
        for neighbour, distance in distances[current].items():

            if neighbour not in unvisited: continue
            newDistance = currentDistance + distance
            if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
                unvisited[neighbour] = newDistance
        visited[current] = currentDistance
        del unvisited[current]
        if not unvisited: break
        candidates = [node for node in unvisited.items() if node[1]]
        current, currentDistance = sorted(candidates, key = lambda x: x[1])[0]
    return visited
 
nodes = ('A', 'B', 'C', 'D', 'E', 'F', 'G')
distances = {
    'A': {'B': 5, 'D': 3, 'E': 12, 'F' :5},
    'B': {'A': 5, 'D': 1, 'G': 2},
    'C': {'G': 2, 'E': 1, 'F': 16},
    'D': {'B': 1, 'G': 1, 'E': 1, 'A': 3},
    'E': {'A': 12, 'D': 1, 'C': 1, 'F': 2},
    'G': {'B': 2, 'D': 1, 'C': 2},
    'F': {'A': 5, 'E': 2, 'C': 16}}
 
print(dijkstra(nodes, distances))