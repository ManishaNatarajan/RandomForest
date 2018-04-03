from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self, max_depth =30   , min_size=1):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = {}
        self.max_depth = max_depth
        self.min_size = min_size
        pass

    def find_max(self, dataset):
        counts = np.bincount(dataset[:,-1].astype(int).reshape(-1,))
        return np.argmax(counts)

    
    def find_split(self, dataset):


        info_gain = float("-inf");
        split_attribute = 0;
        split_value = 0;
        for split_index in range(len(dataset[0,:-1])):
            
            if(dataset[:,split_index].dtype.type is np.string_):
                (_, idx, counts) = np.unique(dataset[:, split_index], return_index=True, return_counts=True)
                index = idx[np.argmax(counts)]
                mode = dataset[index, split_index]
                split_value1 = mode

            else:
                num_med = np.median(dataset[:, split_index])
                split_value1 = num_med

            X_left, X_right, y_left, y_right = partition_classes(dataset[:,:-1], dataset[:,-1].reshape(-1,1), split_index, split_value1)
            info_gain_new = information_gain(dataset[:,-1].astype(int), [y_left.astype(int), y_right.astype(int)])

            if(info_gain_new > info_gain):
                info_gain = info_gain_new
                split_attribute = split_index
                split_value = split_value1

        return {'split_attribute' : split_attribute, 'split_value': split_value, 'partitioned_data': (np.concatenate((X_left,y_left.reshape(-1,1)), axis=1), np.concatenate((X_right, y_right.reshape(-1,1)), axis =1))}

    

    def split(self, node, max_depth, min_size, cur_depth):
        left, right = node['partitioned_data']
        del(node['partitioned_data'])

        if(left.size == 0 or right.size == 0):
            node['left'] = node['right'] = self.find_max(np.concatenate((left,right), axis =0))
            return

        if(cur_depth > max_depth):
            node['left'], node['right'] = self.find_max(left), self.find_max(right)
            return

        if(len(left)<min_size):
            node['left'] = self.find_max(left)
        else:
            node['left'] = self.find_split(left)
            self.split(node['left'], max_depth, min_size, cur_depth+1)
        if(len(right)<min_size):
            node['right'] = self.find_max(right)
        else:
            node['right'] = self.find_split(right)
            self.split(node['right'], max_depth, min_size, cur_depth+1)

    def traverse(self, node, record):

        if(isinstance(record[node['split_attribute']], int)):

            if(record[node['split_attribute']]<=node['split_value']):
                if(isinstance(node['left'], dict)):
                    return self.traverse(node['left'],record)
                else:
                    return node['left']

            elif(record[node['split_attribute']] > node['split_value']):
                if(isinstance(node['right'], dict)):
                    return self.traverse(node['right'], record)
                else:
                    return node['right']
        else:

            if(record[node['split_attribute']]==node['split_value']):
                if(isinstance(node['left'], dict)):
                    return self.traverse(node['left'],record)
                else:
                    return node['left']

            elif(record[node['split_attribute']] != node['split_value']):
                if(isinstance(node['right'], dict)):
                    return self.traverse(node['right'], record)
                else:
                    return node['right']


    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
       
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        self.X = X
        self.y = y
        self.tree = self.find_split(np.concatenate((self.X, self.y), axis = 1))
        self.split(self.tree, self.max_depth, self.min_size, 1 )
        pass


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label

        return self.traverse(self.tree, record)
        
