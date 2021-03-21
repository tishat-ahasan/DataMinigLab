import numpy as np
import pandas as pd
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
#     print(dataset.head(5))
    return attributes, dataset

class NaiveBayesClassifier:
    def __init__(self, training_data, attributes):
        self.training_data = training_data
        self.attributes = attributes
        self.Info = self.getInfo(self.training_data, self.attributes)

    def getInfo(self,dataset, attributes):
        Info = {}
        mean = {}
        std = {}
    #     grouped = dataset.group_by(dataset['class'])
        for column in dataset.columns:
            if column == 'class' or attributes[column] == 'category': continue
            mean[column] = dataset.groupby('class')[column].mean().to_dict()
            std[column] = dataset.groupby('class')[column].std().to_dict()
        Info['mean'] = mean
        Info['std'] = std
        return Info


    def getPrediction(self, dataset, Info, x):
        distinct_class = dataset['class'].value_counts()
        classProb = distinct_class/ distinct_class.sum()
        grouped = dataset.groupby(['class'])
        Winner = None
        maxPosterior = -np.inf
        for att_class in distinct_class.index:
            like_hood = 0
            OnlyClassData = grouped.get_group(att_class)
            for column in dataset.columns:
                if column == 'class': continue
                if attributes[column] == 'category':
                    grouped_column = (OnlyClassData.groupby(column).count())/len(OnlyClassData)
                    if x[column] in grouped_column['class'].index:
                        conditionalProbability = np.log(grouped_column['class'][x[column]])
                    else: 
                        conditionalProbability = np.log(1e-6)
                    like_hood += conditionalProbability
                else:
                    conditionalProbability = self.normal_PDF(x[column],self.Info['mean'][column][att_class],self.Info['std'][column][att_class])
                    conditionalProbability += 1e-6
                    like_hood += np.log(conditionalProbability)
            posterior = like_hood+np.log(classProb[att_class])
            if posterior > maxPosterior: 
                maxPosterior = posterior
                Winner = att_class
        return Winner

    def normal_PDF(self, val, mu, sigma):
        sigma = sigma if sigma != 0 else self.eps 
        exponentTerm = (-1) * ( ( (val-mu) ** 2 ) / ( 2 * (sigma ** 2) ) )
        return (1/(np.sqrt(2*np.pi) * sigma)) * np.exp(exponentTerm)
    
    def predict(self, XTest):
        YPred = []
        for index,row in XTest.iterrows():
#             print(row)
            YPred.append(self.getPrediction(self.training_data, self.Info, row))
        return np.array(YPred) 

def printStatistics(Y,YPred):
    accuracy = accuracy_score(Y, YPred)*100
    precision = precision_score(Y, YPred, average="macro",zero_division=1)*100
    recall = recall_score(Y, YPred, average="macro",zero_division=1)*100
    f1 = f1_score(Y, YPred, average="macro",zero_division=1)*100
    
    print("Accuracy:","{:.2f}".format(accuracy),"%")
    print("Precision:","{:.2f}".format(precision),"%")
    print("Recall:","{:.2f}".format(recall),"%")
    print("F Measure:","{:.2f}".format(f1),"%")


dataset_name = 'iris'
attributes, dataset = getData(dataset_name)
print("Data Loading Successfull!!!")
print("Dataset:",dataset_name)
print("\n")
testSize = 0.2
training_data, testing_data = train_test_split(dataset, test_size = testSize)
naiveBayesClassifier = NaiveBayesClassifier(training_data, attributes)
YPred = naiveBayesClassifier.predict(testing_data)
printStatistics(list(testing_data['class']), YPred)
