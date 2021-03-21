import math
import resource
import tracemalloc
import time

class TrieNode:
    def __init__(self, item):
        self.item = item
        self.is_end = False
        self.counter = 0
        self.children = {}
        
class Trie(object):

    def __init__(self):
        self.root = TrieNode("root")
        
    def initial_insert(self,item_list,MIN_COUNT):
        for key in sorted(item_list):
            if item_list[key] < MIN_COUNT:  # non frequent item
                continue
            new_node = TrieNode(key)
            new_node.counter = item_list[key]
            new_node.is_end = True
            self.root.children[str(key)] = new_node
    
    def insert(self, item_set):
        node = self.root
        for item in item_set:
            if item in node.children:
                node = node.children[item]
            else:
                new_node = TrieNode(item)
                node.children[item] = new_node
                node = new_node
        node.is_end=True
    
    def generate_candidate(self,root): # Last node ta ki kora jay!!..............
        if root.is_end: return
        keys = list(root.children.keys())
        for i in range (len(keys)):
            cur_node = root.children[keys[i]] 
            if cur_node.is_end == True:
                for j in range(i+1,len(keys)):
                    new_node = TrieNode(root.children[keys[j]].item)
                    new_node.is_end = True
                    cur_node.is_end = False
                    cur_node.children[keys[j]] = new_node
            else:
                self.generate_candidate(cur_node)
    
    def print_trie(self,root,level):
        if root.is_end == True:
            print(level,"leave -->", root.item)
            print(level,"count:",root.counter)
            return
        else:
            print(level,"childer of------>",root.item)
            print(level,"count:",root.counter)
            level = level+"  "+level
            for child in root.children.values():
                self.print_trie(child,level)
                
    def frequency_count(self,root,transaction_list,left):
        if root.is_end == True:
            root.counter += 1
            return
        for child in root.children.values():
            if child.item in transaction_list[left:]:
                offset = transaction_list[left:].index(child.item)
#                 self.frequency_count(child,transaction_list,left)
                self.frequency_count(child,transaction_list,left+offset)
        return
    
    def return_count(self,root,candidate,support_count):
        if root.is_end == True:
            if root.item!= 'root': 
                support_count[candidate] = root.counter
                #print(candidate,":",root.counter)
#             pri#nt(root.item)
            return
        if (len(candidate)!=0): candidate += ","
        for child in root.children.values():
            abc = self.return_count(child,candidate+str(child.item),support_count)

                
    def remove_nonfrequent_item(self,root,min_item,level,cur_level):
        if (root.is_end):
            if (cur_level < level or root.counter < min_item): return True
            else: return False
        delete = [key for key,value in root.children.items() if (self.remove_nonfrequent_item(value,min_item,level,cur_level+1))]
        for key in delete: 
            del root.children[key]
        if (len(root.children)==0): 
            root.is_end = True
            return True
        return False
    
    def database_scan(self,transactions):
        for transaction in transactions:
            self.frequency_count(self.root,transaction,0)


def get_data(filename):
    file = open(filename, 'r')
    data = []
    for line in file:
        row = line.strip().split(' ')
        row = [int(x) for x in row]
        row.sort()
        data.append(row)
        
        
    file.close()
    return data
def item_count(transactions):
    item_list = {}
    for transaction in transactions:
        for item in transaction:
            # key = int(str(item))
            key = item
            item_list[key] = item_list.get(key, 0) + 1
    return item_list

def run(item_list,transactions,MIN_COUNT,trie,FREQUENT_PATTERN):
    batch_no = 1
    while True:
        if batch_no==1:
           trie.initial_insert(item_list,MIN_COUNT) 
          #  trie.print_trie(trie.root,"")
        else:
            trie.generate_candidate(trie.root)
    #         trie.remove_nonfrequent_item(trie.root,0,batch_no,0)
            trie.database_scan(transactions)
            #print("............................")
            #trie.return_count(trie.root,"",FREQUENT_PATTERN) 
            #trie.return_count()
            trie.remove_nonfrequent_item(trie.root,MIN_COUNT,batch_no,0)
        if trie.root.is_end == True: break
        
        trie.return_count(trie.root,"",FREQUENT_PATTERN) 
        batch_no += 1
    return

def Apriori(fname , min_support):
    transactions = get_data(fname)
    item_list = item_count(transactions)
    total_transactions = len(transactions)
    MIN_COUNT = math.ceil(total_transactions*min_support) 
    FREQUENT_PATTERN = {}
    trie = Trie()
    run(item_list,transactions,MIN_COUNT,trie,FREQUENT_PATTERN)
    return FREQUENT_PATTERN

def main():
    print("Generating patterns using Apriori....")
    fname = 'Data/mushroom.txt'
    tracemalloc.start()
    S_Time = time.process_time()
    min_sup = 0.3
    fp = Apriori(fname,min_sup)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    E_TIME = round(time.process_time()-S_Time,4)
    MEMORY = round(peak/10**6,4)
    dash = '-' * 90
    print(dash)
    print('{:<25s}{:>8s}{:>18s}{:>16s}{:>17s}'.format('Dataset', 'min_sup', 'patterns', 'Runtime' , 'Memory'))
    print(dash)
    print('{:<25s}{:>6.2f}%{:>16d}{:>18.3f}{:>18.3f}'.format(fname,min_sup*100,len(fp),E_TIME,MEMORY))
    # print(dash)
    # print("\nGenerated patterns:")
    # for key,value in fp.items():
    # 	print(key,"\t",value)
    


if __name__ == '__main__':
    main()
