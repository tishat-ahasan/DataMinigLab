import pandas as pd
import numpy as np
import math
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('always')

def getData(dataset_name):
    attribute_file_name = 'Data/'+dataset_name+".attribute"
    dataset_file_name = 'Data/'+dataset_name+".data"
    att = pd.read_csv(attribute_file_name,
                      delim_whitespace=True,
                     header = None)
    attributes = {rows[0]:rows[1] for _,rows in att.iterrows()}
    dataset = pd.read_csv(dataset_file_name,
                      names=list(attributes.keys()))
    return attributes, dataset


def getEntropy(target_col, col_type, split_point=0.0):
    if col_type == 'category':
        counts = list(target_col.value_counts().values)
    else:
        left = target_col <= split_point
        right = target_col > split_point
        counts = [len(target_col[left]), len(target_col[right])]
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(counts))])
    return entropy

def InfoGainRatio(data,split_attribute_name,split_att_type,target_name="class"):
    total_entropy = getEntropy(data[target_name], attributes[target_name])
    

    if split_att_type == 'category':
        tmp = data[split_attribute_name].value_counts()
        val = list(tmp.index)
        counts = list(tmp.values)
        information = data[split_att_type].value_counts()
        Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*getEntropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name], attributes[split_attribute_name]) for i in range(len(vals))])
        Information_Gain = total_entropy - Weighted_Entropy
        if Information_Gain == 0.0:
            return Information_Gain
        Gain_Ratio = Information_Gain/getEntropy(data[split_attribute_name], attributes[split_attribute_name])
        return Gain_Ratio, None
    else:
        values = list(np.unique(data[split_attribute_name]))
        best = 0
        idx = None
        for val in values:
            left = data[split_attribute_name] <= val
            right = data[split_attribute_name] > val
            counts = [len(data[split_attribute_name][left]), len(data[split_attribute_name][right])]
            Weighted_Entropy = (counts[0]/np.sum(counts))*getEntropy(data.where(data[split_attribute_name]<=val).dropna()[target_name], attributes[target_name], val) + (counts[1]/np.sum(counts))*getEntropy(data.where(data[split_attribute_name]>val).dropna()[target_name], attributes[target_name], val)
            Information_Gain = total_entropy - Weighted_Entropy
            if Information_Gain == 0.0:
                continue
            Gain_Ratio = Information_Gain/getEntropy(data[split_attribute_name], attributes[split_attribute_name], val)
            if Gain_Ratio>=best:
                best = Gain_Ratio
                idx = val
        return best, idx

def makeDecisionTree(data,features,target_attribute_name="class",parent_node_class = None):
    if len(features)==0 or len(data) == 0 or len(data.columns) == 1:
        return parent_node_class
    try:
        if len(np.unique(data[target_attribute_name])) <= 1:
            return np.unique(data[target_attribute_name])[0]
    except KeyError:
        print("Key Error")
    else:
        parent_node_class = data[target_attribute_name].value_counts().idxmax()
        max_GR = -math.inf 
        for feature in features:
            GR, point = InfoGainRatio(data,feature,target_attribute_name)
            if GR> max_GR:
                max_GR = GR
                split_point = point
                best_feature = feature
        tree = {best_feature:{}}
        features = features[features != best_feature]
        if attributes[best_feature] == 'category':
            grouped = data.groupby(data[best_feature])
            for value in np.unique(data[best_feature]):
                sub_data = grouped.get_group(value)
                if best_feature != target_attribute_name:
                    del sub_data[best_feature]
                subtree = makeDecisionTree(sub_data,features,target_attribute_name,parent_node_class)
                tree[best_feature][value] = subtree
            return(tree)
        else:
            sub_data1 = data[data[best_feature]<=split_point]
            sub_data2 = data[data[best_feature]>split_point]
            if best_feature != target_attribute_name:
                del sub_data1[best_feature]
                del sub_data2[best_feature]
            subtree1 = makeDecisionTree(sub_data1,features,target_attribute_name,parent_node_class)
            subtree2 = makeDecisionTree(sub_data2,features,target_attribute_name,parent_node_class)
            tree[best_feature][split_point] = [subtree1, subtree2]
            return(tree)


def predict(query,tree,default = 1):
    if not isinstance(tree, dict):
        return tree
    att_name = list(tree.keys())[0]
    if attributes[att_name] == 'category':
        try:
            result_tree = tree[att_name][query[att_name]]
        except:
            return default
        result_tree = tree[att_name][query[att_name]]
        return predict(query, result_tree)
    else:
        key_val = list(tree[att_name].keys())[0]
        if  query[att_name]<=key_val:
            result_tree = tree[att_name][key_val][0]
        else:
            result_tree = tree[att_name][key_val][1]
        return predict(query, result_tree)

def printDecisionTree(tree,level):
    level += " "
    for key,value in tree.items():
        if isinstance(value, dict):
            print(level+str(key)+":")
            printDecisionTree(value, level)
        elif isinstance(value,list):
            if isinstance(value[0],dict):
                print(level+str(key)+">= (less than):")
                printDecisionTree(value[0], level)
            else:
                print(level+str(key)+">= (less than):"+"-->",value[0])
                
            if isinstance(value[1],dict):
                print(level+str(key)+" < (greater than):")
                printDecisionTree(value[1], level)
            else:
                print(level+str(key)+"< (greater than):"+"-->",value[1])
        else: 
            print(level+str(key)+"-->",value)    


def test(data,tree, features):
    original_data = list(data['class'])
    queries = data[features].to_dict(orient = "records")
    predictions = [predict(query,tree) for query in queries]
    accuracy = accuracy_score(original_data, predictions)
    precision = precision_score(original_data, predictions, average="macro")
    recall = recall_score(original_data, predictions, average="macro")
    f1 = f1_score(original_data, predictions, average="macro")
    return accuracy*100, precision*100, recall*100, f1*100

attributes, dataset = getData(dataset_name='sample')
print("Data Load successfully!!!")
testSize = 0.2
training_data, testing_data = train_test_split(dataset, test_size = testSize)
# print(attributes)
features = training_data.columns
features = features[features!= 'class']

tree = makeDecisionTree(training_data,features)
accuracy, precision, recall, f1 = test(testing_data,tree, features)

#         print('DecisionTree,'+str(testSize)+","+str(accuracy)+","+str(precision)+","+str(recall)+","+str(f1))

print("Accuracy:","{:.2f}".format(accuracy),"%")
print("Precision:","{:.2f}".format(precision),"%")
print("Recall:","{:.2f}".format(recall),"%")
print("F Measure:","{:.2f}".format(f1),"%")
