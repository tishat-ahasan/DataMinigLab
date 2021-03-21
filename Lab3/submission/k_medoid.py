import pandas as pd
import numpy as np
import random
import math
import copy
from statistics import mean 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  silhouette_score
from matplotlib import pyplot as plt
# random.seed(5)


def getData(dataset_name):
    attribute_file_name = 'Data/'+dataset_name+".attribute"
    dataset_file_name = 'Data/'+dataset_name+".data"
    att = pd.read_csv(attribute_file_name,
                      delim_whitespace=True,
                     header = None)
    attributes = {rows[0]:rows[1] for _,rows in att.iterrows()}
    dataset = pd.read_csv(dataset_file_name,
                      names=list(attributes.keys()))
    if 'class' in attributes: 
        tuple_labels = []
        label_count = []
        classes = {}
        tuple_labels = list(dataset["class"])
        classes_unique, label_count = np.unique(tuple_labels, return_counts=True)
        idx = 0
        for class_ in classes_unique:
            classes[class_] = idx
            idx += 1
#         print(classes)
        del attributes['class']; del dataset['class']
    return  dataset,classes,tuple_labels,label_count



def dist(x1,x2, minkowski = 2):
    val = 0.0
#     print("In error: ",attributes)
    for att in attributes:
#         if attributes[att]=='value':
        val += (x1[att] - x2[att])*(x1[att] - x2[att])
    val = math.sqrt(val)
    return val


def assignCluster(curr_centroid_index):
    new_cluster_index = [-1]*len(dataset)
    total_error = 0
#     flag = False
    for i in range(len(dataset)):
        min_val = math.inf
        min_idx = -1
        for j in range(len(curr_centroid_index)):
            distance = dist(dataset.iloc[i],dataset.iloc[curr_centroid_index[j]])
            if distance < min_val:
                min_val = distance
                min_idx = j
        new_cluster_index[i] = min_idx
        total_error += abs(min_val)
    return new_cluster_index, total_error

def bcubed():
    if len(classes) == 0:
        print("no labels")
        return
    cluster_label_combo = np.zeros([len(centroid_index), len(classes)])
    cluster_count = np.zeros([len(centroid_index)])
    for i in range(len(dataset)):
        cluster_label_combo[cluster_index[i]][classes[tuple_labels[i]]] += 1.0
        cluster_count[cluster_index[i]] += 1.0

    bcp = 0.0
    bcr = 0.0
    for i in range(len(dataset)):
        bcp += cluster_label_combo[cluster_index[i]][classes[tuple_labels[i]]]/cluster_count[cluster_index[i]]
        bcr += cluster_label_combo[cluster_index[i]][classes[tuple_labels[i]]]/label_count[classes[tuple_labels[i]]]
    bcp /= len(dataset)
    bcr /= len(dataset)
    return bcp,bcr


def showGraph(loop_num):
    x = []
    y = []
    color = ["red","green", "blue", "yellow", "black"]
    for i in range(len(centroid_index)+1):
        x.append([])
        y.append([])
    for i in range(len(dataset)):
        x[cluster_index[i]].append(dataset.iloc[i]['x'])
        y[cluster_index[i]].append(dataset.iloc[i]['y'])
    for i in range(len(centroid_index)):
        x[len(centroid_index)].append(dataset.iloc[centroid_index[i]]['x'])
        y[len(centroid_index)].append(dataset.iloc[centroid_index[i]]['y'])
    for i in range(len(centroid_index)):
        plt.scatter(x[i],y[i],color=color[i])
    # print("Centroids: ")
    
    plt.scatter(x[-1],y[-1],color="black")
    filename = "fig_"+str(loop_num)+".png"
    # plt.savefig(filename)
    plt.show()



dataset_name = 'test'
k = 4
groundTruth = False
if groundTruth:
    dataset,classes,tuple_labels,label_count = getData(dataset_name)
else:
    filepath="Data/"+dataset_name+".data"
    dataset = pd.read_csv(filepath)
dataset = dataset.dropna()
min_max_scaler = MinMaxScaler()
value_attributes = list(dataset.columns)
attributes = list(dataset.columns)
dataset[value_attributes] = min_max_scaler.fit_transform(dataset[value_attributes])

print("_______________________ Datase:",dataset_name,"_____________________")
print("Dataset Size:",dataset.shape)
print("K:",k)
print("\n")
centroid_index=random.sample(range(0,len(dataset)),k)
cluster_index, total_error = assignCluster(centroid_index)
L = 0
while L<10:
    L +=1
    print("Iteration",L,"Variance: ",total_error)
#         print()
    if dataset_name=='test':
        showGraph(L)
    Flag = False
    prev_centroid_index = copy.deepcopy(centroid_index)
    for i in range(k):
        for j in random.sample(range(1,len(dataset)),int(len(dataset)*0.2)):
            if i==j: continue
            new_centroid_index = copy.deepcopy(centroid_index)
            new_centroid_index[i] = j
            new_cluster_index, new_total_error = assignCluster(new_centroid_index)
            if (new_total_error < total_error):
                total_error = new_total_error
                cluster_index = copy.deepcopy(new_cluster_index)
                centroid_index = copy.deepcopy(new_centroid_index)
                Flag = True
                break
    if Flag==False:
        break
print("Execution Completed Successfully!!!")
silhouette = silhouette_score(dataset.values.tolist(),cluster_index)
variance = total_error
print("\n\n")
print("Silhouette Coefficient:",silhouette)
print("Variance:",variance)
if groundTruth:
    precision,recall = bcubed()
    print("BCubed Precision:",precision)
    print("BCubed Recall:",recall)
