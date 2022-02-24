import numpy as np
import matplotlib as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering,Birch,DBSCAN
from ensemble_clustering import relabel_cluster, voting
from sklearn.metrics.cluster import normalized_mutual_info_score,silhouette_score
from Co_assossiation import ComputeCoassossiation
# X, y = load_dataset('julei.csv')
from sklearn.cluster import AgglomerativeClustering as AG
import scipy.io
import networkx as nx
label=[]
with open('LabelProb.txt','r') as ff:
    for line in ff:
        label.append(int(line.rstrip().strip()))
print(label)
# def readdataemail():
#     fdata=open('email-Eu-core.txt','r')
#     X=np.zeros((1005,1005))
#     label=[]
#     for line in fdata:
#         l=line.rstrip().strip()
#         l=l.split(' ')
#         i=int(l[0])
#         j=int(l[1])
#         X[i,j]+=1
#     flabel=open('email-Eu-core-department-labels.txt','r')
#     for line in flabel:
#         l = line.rstrip().strip()
#         l = l.split(' ')
#         lab = int(l[1])
#         label.append(lab)
#     return X,np.array(label),max(label)+1
# X,Y,numcluster=readdataemail()
# G = nx.read_gml('football.gml')
# A=nx.adjacency_matrix(G)
# X=A.todense()
mat = scipy.io.loadmat('matlab.mat')
m=mat['A']
X=m.todense()
numcluster=2
#Step1
y_KMeans = KMeans(n_clusters=numcluster).fit(X)
y_SC = SpectralClustering(n_clusters=numcluster).fit(X)
y_AC = AgglomerativeClustering(n_clusters=numcluster).fit(X)
res1=ComputeCoassossiation(list(y_KMeans.labels_))
res2=ComputeCoassossiation(list(y_SC.labels_))
res3=ComputeCoassossiation(list(y_AC.labels_))
result1=(1/3)*(res1+res2+res3)
####################################################################
y_KMeans = KMeans(n_clusters=numcluster).fit(X)
y_SC = SpectralClustering(n_clusters=numcluster).fit(X)
y_AC =DBSCAN(eps=0.3, min_samples=10).fit(X)
res1=ComputeCoassossiation(list(y_KMeans.labels_))
res2=ComputeCoassossiation(list(y_SC.labels_))
res3=ComputeCoassossiation(list(y_AC.labels_))
result2=(1/3)*(res1+res2+res3)
#########################################################3
y_KMeans = AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=numcluster).fit(X)
y_SC = SpectralClustering(n_clusters=numcluster).fit(X)
y_AC = AgglomerativeClustering(n_clusters=numcluster).fit(X)
res1=ComputeCoassossiation(list(y_KMeans.labels_))
res2=ComputeCoassossiation(list(y_SC.labels_))
res3=ComputeCoassossiation(list(y_AC.labels_))
result3=(1/3)*(res1+res2+res3)
#######################################
resultfinal=(1/3)*(result1+result2+result3)
# newy=[]
# for j in y:
#     if j == 'Good':
#         newy.append(1)
#     else:
#         newy.append(0)
# print("==========")
# print(newy)
# print("==========")
Y=label
model = SpectralClustering(n_clusters=numcluster)
model.fit(resultfinal)
clusters = []
clusters.append(list(Y))
clusters.append(list(model.labels_))
relabeled_clusters = relabel_cluster(clusters)
print("normalized_mutual_info_score:",normalized_mutual_info_score(model.labels_,Y))
print("silhouette_score:",silhouette_score(np.array(relabeled_clusters[0]).reshape(-1,1),np.array(relabeled_clusters[1]).reshape(-1,1)))
print(list(model.labels_))
print('hello world')