import pandas as pd
import numpy as np
import random
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  silhouette_score
from statistics import mean 
from matplotlib import pyplot as plt



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
    for att in dataset.columns:
        val += float(x1[att] - x2[att])*float(x1[att] - x2[att])
    return val

def calculateError():
    error = 0
    sqrt_error = 0
    quality = 0
    for i in range(len(dataset)):
        distance = dist(dataset.iloc[i], pd.DataFrame(centroids[cluster_index[i]],index=[0]))
        error += distance
    return error

def assignCluster():
    flag = False
    for i in range(len(dataset)):
        min_val = math.inf
        min_idx = -1
        for j in range(len(centroids)):
            distance = dist(dataset.iloc[i], pd.DataFrame(centroids[j],index=[0]))
            if distance < min_val:
                min_val = distance
                min_idx = j
        if flag==False and cluster_index[i] != min_idx:
            print
            flag = True
        cluster_index[i] = min_idx
    return flag

def newCentroids():
    for i in range(len(centroids)):
        List = [j for j in range(len(dataset)) if cluster_index[j]==i]
        clustered_data = dataset.iloc[List]
        if (len(clustered_data)>0):
            for column in dataset.columns:
                centroids[i][column] = clustered_data[column].mean()


def bcubed():
    if len(classes) == 0:
        print("no labels")
        return
    cluster_label_combo = np.zeros([len(centroids), len(classes)])
    cluster_count = np.zeros([len(centroids)])
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
    for i in range(len(centroids)+1):
        x.append([])
        y.append([])
    for i in range(len(dataset)):
        x[cluster_index[i]].append(dataset.iloc[i]['x'])
        y[cluster_index[i]].append(dataset.iloc[i]['y'])
    for i in range(len(centroids)):
        x[len(centroids)].append(centroids[i]['x'])
        y[len(centroids)].append(centroids[i]['y'])
    for i in range(len(centroids)):
        plt.scatter(x[i],y[i],color=color[i])
    # print("Centroids: ")
    # print(x[-1],y[-1])
    
    plt.scatter(x[-1],y[-1],color="black")
    filename = "fig_"+str(loop_num)+".png"
#     print(filename)
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
dataset[value_attributes] = min_max_scaler.fit_transform(dataset[value_attributes])


print("_______________________ Datase:",dataset_name,"_____________________")
print("Dataset Size:",dataset.shape)
print("K:",k)
print("\n")
centroids = []
cluster_index = [-1]*len(dataset)
for i in range (k): 
    centroid = {}
    random_number = random.randint(0,len(dataset)-1)
    for column in dataset.columns:
        centroid[column] = dataset.iloc[random_number][column]
    centroids.append(centroid)
loop_num = 0

while True:
    loop_num += 1
    continue_loop = assignCluster()
    newError = calculateError()
    if dataset_name=='test':
        showGraph(loop_num)
    print("Iteration",loop_num,"Variance: ",newError)
    newCentroids()
    if continue_loop == False or loop_num>20:
        break

print("Execution Completed Successfully!!!")
silhouette = silhouette_score(dataset.values.tolist(),cluster_index)
variance = newError
print("\n\n")
print("Silhouette Coefficient:",silhouette)
print("Variance:",variance)
if groundTruth:
    precision,recall = bcubed()
    print("BCubed Precision:",precision)
    print("BCubed Recall:",recall)
