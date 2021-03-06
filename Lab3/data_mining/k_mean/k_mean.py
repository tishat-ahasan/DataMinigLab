import pandas as pd
import numpy as np
from pprint import pprint
import random
import time
import copy
import math
from matplotlib import pyplot as plt

# for i in range(25):
#     x = random.random()*5
#     y = random.random()*5
#     print(str(x) + "," + str(y))
# for i in range(25):
#     x = random.random()*5 + 5
#     y = random.random()*5 + 5
#     print(str(x) + "," + str(y))
# for i in range(25):
#     x = random.random()*5 + 10
#     y = random.random()*5 + 10
#     print(str(x) + "," + str(y))
# for i in range(25):
#     x = random.random()*5 + 15
#     y = random.random()*5 + 15
#     print(str(x) + "," + str(y))
color = ["red","green", "blue", "yellow", "black"]

inf = 1e10
def read_file_to_dict(filename):
    att_dict = {}
    f = open(filename, "r")
    i = 1
    for line in f:
        words = line.split()
        att_dict[words[0]] = words[1]
        # print(i, words, len(att_dict))
        i += 1
    return att_dict

attributes = read_file_to_dict("../test_att.txt")
dataset = pd.read_csv('../test.data',
                      names=list(attributes.keys()))


# dataset = dataset.sample(frac=1).reset_index(drop=True)
if "class" in attributes:
    attributes.pop("class")
    dataset = dataset.drop(["class"], axis=1)

# atutu = list(attributes.keys())
# atutu.remove("class")
# atutu.append("class")
# dataset = dataset.reindex(columns=atutu)

for att in attributes:
    min_val = min(dataset[att])
    max_val = max(dataset[att])
    dataset[att] = dataset[att].apply(lambda x: x - min_val )
    dataset[att] = dataset[att].apply(lambda x: x/(max_val-min_val))


tuples = dataset.iloc[:].to_dict(orient = "records")


k = 4 # initialize k
init_centroid_idx = random.sample(range(0, len(dataset)), k)
# print(init_centroid_idx)
init_centroid_idx = [99, 98, 97 ,96]
# print(init_centroid_idx)
centroids = []
cluster_id = []
for i in range(len(tuples)):
    cluster_id.append(i)
for i in range(k):
    centroids.append(copy.deepcopy(tuples[init_centroid_idx[i]]))

def dist(x,y, minkowski = 2):
    val = 0.0
    for att in attributes:
        val += (x[att] - y[att])*(x[att] - y[att])
    val = math.sqrt(val)
    return val

def calc_error():
    err = np.zeros([len(centroids)])
    err2 = np.zeros([len(centroids)])
    err_check = 0
    for i in range(len(tuples)):
        err_check += dist(tuples[i], centroids[cluster_id[i]])**2
        err2[cluster_id[i]] += dist(tuples[i], centroids[cluster_id[i]])**2
        err[cluster_id[i]] += dist(tuples[i], centroids[cluster_id[i]])
    print("Erroooorrrr.............")
    print(np.sum(err), np.sum(err2), err_check)

    return np.sum(err2)

def assign_cluster():
    for i in range(len(tuples)):
        min_val = inf
        min_idx = -1
        for j in range(len(centroids)):
            if dist(tuples[i], centroids[j]) < min_val:
                min_val = dist(tuples[i], centroids[j])
                min_idx = j
        cluster_id[i] = min_idx

def new_means():
    for i in range(len(centroids)):
        for att in attributes:
            centroids[i][att] = 0.0
    count = np.zeros([len(centroids)])
    for i in range(len(tuples)):
        count[cluster_id[i]] += 1
        for att in attributes:
            centroids[cluster_id[i]][att] += tuples[i][att]
    # print(count)
    for i in range(len(centroids)):
        for att in attributes:
            centroids[i][att] = centroids[i][att] / count[i]

def show_graph():
    x = []
    y = []
    for i in range(len(centroids)+1):
        x.append([])
        y.append([])
    for i in range(len(tuples)):
        x[cluster_id[i]].append(tuples[i]['x'])
        y[cluster_id[i]].append(tuples[i]['y'])
    for i in range(len(centroids)):
        x[len(centroids)].append(centroids[i]['x'])
        y[len(centroids)].append(centroids[i]['y'])
    for i in range(len(centroids)):
        plt.scatter(x[i],y[i],color=color[i])
    print(x[-1],y[-1])
    plt.scatter(x[-1],y[-1],color="black")
    plt.show()

iter_lim = 20
iter = 0
error = inf
while iter<iter_lim:
    # print(centroids)
    assign_cluster()
    show_graph()
    new_error = calc_error()
    # print(new_error)
    if error == new_error:
        break
    error = new_error
    new_means()
    iter += 1
print(cluster_id)
