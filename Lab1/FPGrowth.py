from collections import defaultdict, OrderedDict
import pandas as pd
from itertools import chain, combinations
from optparse import OptionParser
import operator
import resource
import tracemalloc
import time


class Node:
    def __init__(self, itemName, frequency, parentNode):
        self.itemName = itemName
        self.count = frequency
        self.parent = parentNode
        self.children = {}
        self.next = None

    def display(self, ind=1):
        print('  ' * ind, self.itemName, ' ', self.count)
        for child in list(self.children.values()):
            child.display(ind+1)

def get_data(filename):
    transactions = []
    frequency = []
    
    file = open(filename, 'r')
    for line in file:
        row = line.strip().split(' ')
        transactions.append(row)
        frequency.append(1)
    file.close()
    return transactions, frequency

def constructTree(transactions, frequency, minSup):
    item_set = defaultdict(int)
    # Counting frequency and create header table
    for idx, transaction in enumerate(transactions):
        for item in transaction:
            item_set[item] += frequency[idx]

    
    frequent_items = {}
    for key,value in item_set.items():
        if value >= minSup:
            frequent_items[key] = value
    del(item_set)

    frequent_items = dict( sorted(frequent_items.items(), key=operator.itemgetter(1),reverse=True))

    headerTable = defaultdict(int)
    for key,value in frequent_items.items():
#         print("key:",key,"value:",value)
        headerTable[key] = [value, None]
    del(frequent_items)
    
    # Init Null head node
    fpTree = Node('Null', 1, None)
    # Update FP tree for each cleaned and sorted itemSet
    for idx, itemSet in enumerate(transactions):
        itemSet = list(filter(lambda v: v in headerTable, itemSet))
#         itemSet = [item for item in itemSet if item in headerTable]
        itemSet.sort(key=lambda item: headerTable[item][0], reverse=True)
        # Traverse from root to leaf, update tree with given item
        currentNode = fpTree
        for item in itemSet:
            currentNode = updateTree(item, currentNode, headerTable, frequency[idx])
    return fpTree, headerTable

def updateTree(item, treeNode, headerTable, frequency):
    if item in treeNode.children:
        # If the item already exists, increment the count
        treeNode.children[item].count += frequency
    else:
        # Create a new branch
        newItemNode = Node(item, frequency, treeNode) #item,frequency,parent_node
        treeNode.children[item] = newItemNode
        # Link the new branch to header table
        updateHeaderTable(item, newItemNode, headerTable)
    return treeNode.children[item]

def updateHeaderTable(item, targetNode, headerTable):
    if(headerTable[item][1] == None):
        headerTable[item][1] = targetNode
    else:
        currentNode = headerTable[item][1]
        # Traverse to the last node then link it to the target
        while currentNode.next != None:
            currentNode = currentNode.next
        currentNode.next = targetNode


def mineTree(headerTable, minSup, preFix, freqItemList):
    for key,value in headerTable.items():  
        # Pattern growth is achieved by the concatenation of suffix pattern with frequent patterns generated from conditional FP-tree
        if preFix == "": newPattern = preFix + key
        else: newPattern = preFix + "," + key
        freqItemList[newPattern] += value[0]
        # Find all prefix path, constrcut conditional pattern base
        conditionalPattBase, frequency = findPrefixPath(key, headerTable) 
        # Construct conditonal FP Tree with conditional pattern base
        conditionalTree, newHeaderTable = constructTree(conditionalPattBase, frequency, minSup) 
        if newHeaderTable != None:
            # Mining recursively on the tree
            mineTree(newHeaderTable, minSup,
                       newPattern, freqItemList)

def findPrefixPath(basePat, headerTable):
    # First node in linked list
    treeNode = headerTable[basePat][1] 
    condPats = []
    frequency = []
    while treeNode != None:
        prefixPath = []
        # From leaf node all the way to root
        climbFPTree(treeNode, prefixPath)  
        if len(prefixPath) > 1:
            # Storing the prefix path and it's corresponding count
            condPats.append(prefixPath[1:])
            frequency.append(treeNode.count)
        # Go to next node
        treeNode = treeNode.next    
    return condPats, frequency

def climbFPTree(node, prefixPath):
    if node.parent != None:
        prefixPath.append(node.itemName)
        climbFPTree(node.parent, prefixPath)

def fpgrowth(fname, minSupport):
    transactions, frequency = get_data(fname)
    MIN_COUNT = len(transactions) * minSupport
    fpTree, headerTable = constructTree(transactions, frequency, MIN_COUNT)
    del(transactions)
    freqItems = defaultdict(int)
    mineTree(headerTable, MIN_COUNT, "", freqItems)
    return freqItems


def main():
    print("Generating patterns using FP Growth....")
    fname = 'Data/mushroom.txt'
    tracemalloc.start()
    S_Time = time.process_time()
    min_sup = 0.3
    fp = fpgrowth(fname,min_sup)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    E_TIME = round(time.process_time()-S_Time,4)
    MEMORY = round(peak/10**6,4)

    dash = '-' * 90
    print(dash)
    print('{:<25s}{:>8s}{:>18s}{:>16s}{:>17s}'.format('fname', 'min_sup', 'patterns', 'Runtime' , 'Memory'))
    print(dash)
    print('{:<25s}{:>6.2f}%{:>16d}{:>18.3f}{:>18.3f}'.format(fname,min_sup*100,len(fp),E_TIME,MEMORY))
    # print("\nGenerated patterns:")
    # for key,value in fp.items():
    #   print(key,"\t",value)

if __name__ == '__main__':
    main()