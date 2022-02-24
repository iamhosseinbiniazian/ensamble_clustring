import networkx as nx
import matplotlib.pyplot as plt
# G = nx.path_graph(4)
#
# nx.write_gml(G, 'test.gml')
label=[]
with open('Label.txt','r') as ff:
    for line in ff:
        label.append(int(line.rstrip().strip()))
print(label)
# G = nx.read_gml('football.gml')
# nx.draw(G)
# plt.show()
A=nx.adjacency_matrix(G)
print(A.todense())