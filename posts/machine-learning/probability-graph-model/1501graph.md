<script> MathJax.Hub.Queue(["Typeset", MathJax.Hub]); </script>

# 그래프(graph) 기초

### summary

- 그래프는 노드와 간선으로 이루어져 있다. 
- 두 간선을 다른 것으로 본다면 방향성 그래프, 같은 것으로 본다면 비방향성 그래프이다. 
- 워크는 어떤 노드를 출발해서 다른 노드로 도달하기 위한 인접한 노드의 순서열. 가장 넓은 개념이다. 패스는 워크 중에서 시작과 끝을 제외한 다른 노드에 대해서 동일한 노드를 두 번 이상 지나지 않는 워크이다. 사이클은 시작점과 끝점이 동일한 패스, 어사이클릭 그래프(acyclic graph)는 사이클이 없는 그래프이다.
- 무방향성 그래프의 노드 집합 중에서 모든 노드끼리 간선이 존재하면 그 노드 집합을 클리크라고 한다. 만약 클리크에 포함된 노드에 인접한 다른 노드를 추가하면 클리크가 아니게 되는 것을 최대 클리크(maximal clique)라고 한다.
___________________


### 그래프 graph

그래프는 노트(node, vertex)와 그 사이를 잇는 간선(edge)으로 이루어진 구조를 말한다. 수학적으로 그래프 $$G$$ 는 노드(vertex) 집합 $$V$$ 와 간선(edge) 집합 $$E$$ 로 구성된다. (튜플 형식)

$$
G(V, E)
$$

간선은 두 개의 노드로 이루어진 순서가 있는 쌍(ordered pair)이다. 

$$
E \subseteq V x V
$$

### 방향성 그래프(directed graph)와 비방향성 그래프(undirected graph)

만약 간선 (a,b)와 (b,a)이 있을 때 두 간선을 다른 것으로 본다면 간선의 방향이 있는 방향성 그래프(directed graph)이고 두 간선을 같은 것으로 본다면 간선의 방향이 없는 비방향성 그래프(undirected graph)이다. 노드 집합 $$V$$ 와 간선 집합 $$E$$를 가지는 그래프 $$G$$ 에 포함된 노드의 갯수를 그래프의 크기(cardinality)라고 하며 $$|G|, |V|$$ 로 나타내고 간선의 갯수는  $$|E|$$ 로 나타낸다. 만약 두 노드 a,b를 포함하는 간선 (a, b)가 $$E$$ 안에 존재하면 두 노드는 인접하다(adjacent)고 인접한 두 노드는 서로 이웃(neighbor)이라고 한다.

$$
(a,b) \in E
$$

어떤 노드에서 출발하여 자기 자신으로 바로 돌아오는 간선이 있다면 셀프 루프(self loop)라고 한다. 

### 워크(walk), 패스(path), 사이클(cycle), 트레일(trail)

- 워크는 어떤 노드를 출발해서 다른 노드로 도달하기 위한 인접한 노드의 순서열. 가장 넓은 개념이다.
- 패스는 워크 중에서 시작과 끝을 제외한 다른 노드에 대해서 동일한 노드를 두 번 이상 지나지 않는 워크이다.
- 사이클은 시작점과 끝점이 동일한 패스, 어사이클릭 그래프(acyclic graph)는 사이클이 없는 그래프이다.
- 트레일은 어떠한 노드든 동일한 노드를 두 번 이상 지나지 않는 워크이다.

### 클리크 clique

무방향성 그래프의 노드 집합 중에서 모든 노드끼리 간선이 존재하면 그 노드 집합을 클리크라고 한다. 만약 클리크에 포함된 노드에 인접한 다른 노드를 추가하면 클리크가 아니게 되는 것을 최대 클리크(maximal clique)라고 한다.