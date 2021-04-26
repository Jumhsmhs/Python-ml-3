import os
from Utils import DataLoader
from Model import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

mnist_data = DataLoader('dataset')
tr_data, tr_class_labels, tr_subclass_labels = mnist_data.loaddata()

print(tr_data.shape)

mnist_data.plot_imgs(tr_data,25,True)
pca = PCA(n_components=400, whiten=False)
data = pca.fit_transform(tr_data)

kmeans = KMeans(n_clusters=5,max_iter=5)
kmeans.fit(data,tr_class_labels)
mnist_data.plot_imgs([pca.inverse_transform(x) for x in kmeans.centroids], len(kmeans.centroids))
plt.plot(range(kmeans.iterations),kmeans.loss_per_iteration)
plt.show()

for key,data in list(kmeans.clusters['data'].items()):
    print('Cluster: ',key, 'Label:',kmeans.clusters_labels[key])
    new_data = pca.inverse_transform(data)
    mnist_data.plot_imgs(new_data[:min(25,data.shape[0])],min(25,new_data.shape[0]))

print('[cluster_label,no_occurence_of_label,total_samples_in_cluster,cluster_accuracy]\n',kmeans.clusters_info)
print('Accuracy:',kmeans.accuracy)

# To perform PCA we must first change the mean to 0 and variance to 1 for X using StandardScalar
Clus_dataSet = StandardScaler().fit_transform(X) #(mean = 0 and variance = 1)
from sklearn.decomposition import PCA
# Make an instance of the Model
variance = 0.98 #The higher the explained variance the more accurate the model will remain
pca = PCA(variance)
#fit the data according to our PCA instance
pca.fit(Clus_dataSet)


