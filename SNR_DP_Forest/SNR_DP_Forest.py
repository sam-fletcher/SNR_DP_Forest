''' A class that builds a Differentially Private Random Decision Forest using reliable Signal-to-Noise Ratios.
Assumptions: - That the size of the dataset is public. If not, a small amount of the budget can be used to retrieve an estimate.
             - That the attributes and their domains are public; if they weren't, it wouldn't be possible to query the dataset at all. '''

from collections import Counter, defaultdict
import random
import numpy as np
import math

SNR_MULTIPLIER = 1.0 # in case we want to experiment with different signal-to-noise ratios

class SNR_DP_Forest:  
    def __init__(self, 
                 training_data, # 2D list of the training data where the columns are the attributes, and the first column is the class attribute. ONLY categorical data
                 epsilon # the total privacy budget
                 ):
        self._trees = []

        ''' Some initialization '''
        attribute_values = self.get_domains(training_data)
        class_values = [str(y) for y in list(set([x[0] for x in training_data]))]
        attribute_indexes = [int(k) for k,v in attribute_values.items()]
        dataset_size = len(training_data)
        valid_attribute_sizes = [[int(k),len(v)] for k,v in attribute_values.items()]
        num_trees = len(attribute_indexes)
        average_domain = np.mean( [x[1] for x in valid_attribute_sizes] )

        ''' Calculating minimum supoprt threshold '''
        estimated_support_min_depth = dataset_size / (average_domain**2) # a large number
        estimated_support_max_depth = dataset_size / (average_domain ** (len(attribute_indexes)/2)) # max tree depth is k/2 # a small number
        epsilon_per_tree = epsilon / float(num_trees)
        min_support = len(class_values) * math.sqrt(2) * (1/epsilon_per_tree) * SNR_MULTIPLIER # the minimum support in order for S>N

        while estimated_support_max_depth > min_support: # then we can have more trees
            num_trees += 1
            epsilon_per_tree = epsilon / float(num_trees)
            min_support = len(class_values) * math.sqrt(2) * (1/epsilon_per_tree) * SNR_MULTIPLIER

        while min_support > estimated_support_min_depth and num_trees>1: # then we need to have less trees
            num_trees, valid_attribute_sizes, estimated_support_min_depth = self.reduce_trees(num_trees, valid_attribute_sizes, dataset_size)
            epsilon_per_tree = epsilon / float(num_trees)
            min_support = len(class_values) * math.sqrt(2) * (1/epsilon_per_tree) * SNR_MULTIPLIER

        print("NUM TREES = {} & EPSILON PER TREE = {}".format(num_trees, epsilon_per_tree))

        root_attributes = []
        #print("valid atts and their size: {}".format(valid_attribute_sizes))
        for a in attribute_indexes:
            if a in [x[0] for x in valid_attribute_sizes]:
                root_attributes.append(a) # OR: [index, support, gini]

        for t in range(num_trees):
            #print("unused_root_attributes = "+str(root_attributes))
            if not root_attributes:
                root_attributes = attribute_indexes[:]
            root = random.choice(root_attributes)
            root_attributes.remove(root)
            
            tree = SNR_DP_Tree(attribute_indexes, attribute_values, root, class_values, dataset_size, epsilon_per_tree) 
            num_unclassified = tree.filter_training_data_and_count(training_data, epsilon_per_tree, class_values)
            if num_unclassified > 0:
                print("number of unclassified records = {}".format(num_unclassified))
            tree.prune_tree()
            self._trees.append(tree)

    def get_domains(self, data):
        attr_domains = {}
        transData = np.transpose(data)
        for i in range(1,len(data[0])):
            attr_domains[str(i)] = [str(x) for x in set(transData[i])]
            print("original domain length of categorical attribute {}: {}".format(i, len(attr_domains[str(i)])))
        return attr_domains

    def reduce_trees(self, num_trees, valid_attribute_sizes, dataset_size):
        num_trees -= 1
        largest_attribute = sorted(valid_attribute_sizes, key=lambda x:x[1], reverse=True)[0]
        #print("Removing att{} with domain size {}".format(largest_attribute[0], largest_attribute[1])) 
        new_valids = [ x for x in valid_attribute_sizes if x[0] != largest_attribute[0] ]
        average_domain = np.mean( [x[1] for x in valid_attribute_sizes] ) # average with all attributes
        narrowed_average_domain = np.mean([x[1] for x in valid_attribute_sizes if x[0] in [y[0] for y in new_valids] ]) # average without the big attributes
        estimated_support_squared = dataset_size / (average_domain * narrowed_average_domain)
        return num_trees, new_valids, estimated_support_squared


    def evaluate_accuracy_with_voting(self, records, class_index=0):
        ''' Calculate the Prediction Accuracy of the Forest. '''
        actual_labels = [x[class_index] for x in records]
        predicted_labels = []
        leafs_not_used = 0
        count_of_averages_used = 0
        for rec in records:
            class_value_fractions = defaultdict(list)
            for tree in self._trees:
                node, leaf_not_used = tree._classify(tree._root_node, rec)
                noisy_class_counts = node._noisy_class_counts
                leafs_not_used += leaf_not_used
                support = float(sum([v for k,v in noisy_class_counts.items()]))
                for k,v in noisy_class_counts.items():
                    class_value_fractions[k].append(v/support)
            best_confidences = {}
            for k,lis in class_value_fractions.items():
               best_confidences[k] = max(lis)
            best = [None, 0.0]
            for k,class_best in best_confidences.items():
                if class_best > best[1]:
                    best = [k, class_best]
            average_used = False
            for k,class_best in best_confidences.items():
                if class_best == best[1] and k != best[0]:
                    average_used = True
                    #print("original best: {} vs. contender: {}".format(best, (k,class_best)))
                    orig_average = np.mean(class_value_fractions[best[0]])
                    contender_average = np.mean(class_value_fractions[k])
                    #print("original average: {} vs. contender: {}".format(orig_average, contender_average))
                    if contender_average > orig_average:
                        best = [k, class_best]
            count_of_averages_used += 1 if average_used else 0
            predicted_labels.append(int(best[0]))
        counts = Counter([x == y for x, y in zip(predicted_labels, actual_labels)])
        return float(counts[True]) / len(records),   leafs_not_used / (len(records)*len(self._trees)),   count_of_averages_used / len(records)


class SNR_DP_Tree(SNR_DP_Forest):
    _attribute_values = {}
    _num_classes = 0
    _epsilon = 0.0
    _root_node = None
    _id = 0
    _prunings = None # number of nodes = _id - _prunings
    _pruning_attempts = None

    def __init__(self, attribute_indexes, attribute_values, root_attribute, class_values, dataset_size, epsilon_per_tree):
        self._attribute_values = attribute_values
        self._num_classes = len(class_values)
        self._epsilon = epsilon_per_tree
        self._id = 0
        self._prunings = []

        root = simple_node(None, None, root_attribute, 0, 0, [])
        estimated_support = dataset_size / len(self._attribute_values[str(root_attribute)])
        for value in self._attribute_values[str(root_attribute)]: 
            root.add_child( self._make_children([x for x in attribute_indexes if x!=root_attribute], root, 1, value, estimated_support) )
        self._root_node = root


    def prune_tree(self):
        ''' PRUNING ''' 
        attempts, successes = self._prune(self._root_node, 0, 0)
        self._prunings = successes
        self._pruning_attempts = attempts

    def _make_children(self, candidate_atts, parent_node, current_depth, splitting_value_from_parent, estimated_support):
        ''' Recursively makes all the child nodes for the current node, until a termination condition is met. '''
        self._id += 1
        #print("depth = {}".format(current_depth))
        min_support = self._num_classes * math.sqrt(2) * (1/self._epsilon) * SNR_MULTIPLIER
        #print("min support = {:.2f} vs. estimated support = {:.2f}".format(min_support, estimated_support))
        if not candidate_atts or min_support >= estimated_support: # termination conditions
            return simple_node(parent_node, splitting_value_from_parent, None, current_depth, self._id, None) 
        else:
            new_splitting_attr = random.choice(candidate_atts) # pick the attribute that this node will split on
            current_node = simple_node(parent_node, splitting_value_from_parent, new_splitting_attr, current_depth, self._id, []) # make a new node

            new_support = estimated_support / len(self._attribute_values[str(new_splitting_attr)])
            for value in self._attribute_values[str(new_splitting_attr)]: # for every value in the splitting attribute
                child_node = self._make_children([x for x in candidate_atts if x!=new_splitting_attr], current_node, current_depth+1, value, new_support)
                current_node.add_child( child_node ) # add children to the new node
            return current_node

    def _prune(self, node, attempts, successes):
        if not node:
            return None
        ''' Check if the signal-to-noise ratio is less than SNR_MULTIPLIER: '''
        #print("signal-to-noise: {}".format(node._signal_to_noise))
        if node._signal_to_noise <= SNR_MULTIPLIER or sum([v for k,v in node._noisy_class_counts.items()]) < 1:
            if node._parent_node._parent_node: # the minimum tree with be a root and then leafs on level 2.
                attempts += 1
                successes += node._parent_node.remove_child(node) # a level2 node can remove it's children, but the root can't remove it's children.
        if node._children:
            for child in node._children:
                attempts, successes = self._prune(child, attempts, successes)
        return attempts, successes


    def _find_big_nodes(self, node, support, big_nodes):
        if sum(v for k,v in node._noisy_class_counts.items()) > support:
            big_node.append(node)
        if node._children:
            for child in node._children:
                big_nodes = self._find_big_nodes(child, support, big_nodes)
        return big_nodes

    # recursive function for outputting a description of each node.
    def _print_node(self, node, tree_string):
        tree_string += "{} Lvl {}, ID {}->{}: entering value -> {} # noisy class counts: {} & SNR: {:.2} # splitting attribute {}\n".format(
            '~~~~~~'*node._level, node._level, node._parent_node._id if node._parent_node else '', node._id, 
            node._split_value_from_parent, [(k,v) for k,v in node._noisy_class_counts.items()], node._signal_to_noise, node._splitting_attribute)
        if node._children:
            for child in node._children:
                tree_string = self._print_node(child, tree_string)
        return tree_string


    def print_tree(self):
        return self._print_node(self._root_node, "")

    
    def _filter_record(self, record, node, class_index=0):
        if not node:
            return 1
        if not node._children: # if leaf
            node.increment_class_count(record[class_index])
            return 0
        else:
            rec_val = record[node._splitting_attribute]
            child = None
            for i in node._children:
                if i._split_value_from_parent == rec_val:
                    child = i
                    break
            if child is None: # if the record's value couldn't be found:
                return 1
            return self._filter_record(record, child, class_index)

    def _add_noise_to_counts(self, epsilon, node, class_values, redundant_calls):
        if node._children:
            for child in node._children:
                redundant_calls = self._add_noise_to_counts(epsilon, child, class_values, redundant_calls)
        else:
            redundant_calls += node.make_noisy_class_counts(epsilon, class_values)
        return redundant_calls

    def _sum_counts_for_parents(self, node, redundant_calls, epsilon):
        if node._children:
            redundant_calls += node.add_childrens_class_counts(epsilon)
            for child in node._children:
                redundant_calls = self._sum_counts_for_parents(child, redundant_calls, epsilon)
        return redundant_calls

    def filter_training_data_and_count(self, records, epsilon, class_values):
        ''' Find which leaf each record in the training data fits into. Then add noise to the final count of each class value in each leaf.
            epsilon = the epsilon budget for this tree (each leaf is disjoint, so the budget can be re-used). '''
        num_unclassified = 0
        for rec in records:
            num_unclassified += self._filter_record(rec, self._root_node, class_index=0)
        redundant_leaf_counts = self._add_noise_to_counts(epsilon, self._root_node, class_values, 0)
        #print("number of redundant noisy counts of leafs = {}".format(redundant_leaf_counts))
        redundant_parent_counts = self._sum_counts_for_parents(self._root_node, 0, epsilon)
        #print("number of redundant summed counts for parents = {}".format(redundant_parent_counts))
        return num_unclassified


    def _classify(self, node, record):
        if not node:
            return None
        
        if sum([v for k,v in node._noisy_class_counts.items()]) < 1: # if a node is empty, there's no point going deeper down the tree
            while sum([v for k,v in node._noisy_class_counts.items()]) < 1: 
                node = node._parent_node # traverse up the chain
            #print("now high enough: {} at depth {}".format(node._signal_to_noise, node._level))
            return self._select_confident_node(node)
        if not node._children: # if leaf
            return self._select_confident_node(node)
        else: # if parent
            attr = node._splitting_attribute
            rec_val = record[attr]
            child = None
            for i in node._children:
                if i._split_value_from_parent == rec_val:
                    child = i
                    break
            if child is None: # if the record's value couldn't be found, just return the latest majority value
                return self._select_confident_node(node)
            return self._classify(child, record)           


    def _select_confident_node(self, node):
        ''' Return the node in the rule chain that has the highest confidence. 
        Note that if 2 nodes have the same highest confidence (e.g. 100%), then it's probably the same class value, so we don't care which node we use. '''
        lowest_node_fraction = Counter(node._noisy_class_counts).most_common()[0][1] / float(sum([v for k,v in node._noisy_class_counts.items()]))
        confidences = []
        while node:
            counts = Counter(node._noisy_class_counts).most_common()
            majority_fraction = counts[0][1] / float(sum([v for k,v in node._noisy_class_counts.items()]))
            confidences.append( [node, majority_fraction] )
            node = node._parent_node
        largest = sorted(confidences, key=lambda x:x[1], reverse=True)
        ''' Return the node object with the largest majority_fraction ([0][1] is the fraction). Count the number of times the bottom node is NOT the most confident. '''
        return largest[0][0], 0 if largest[0][1]==lowest_node_fraction else 1 


class simple_node:
    _parent_node = None
    _split_value_from_parent = None
    _splitting_attribute = None
    _level = None
    _id = None
    _children = None
    _class_counts = None
    _noisy_class_counts = None
    _signal_to_noise = 0.0

    def __init__(self, parent_node, split_value_from_parent, splitting_attribute, tree_level, id, children):
        self._parent_node = parent_node
        self._split_value_from_parent = split_value_from_parent
        self._splitting_attribute = splitting_attribute
        self._level = tree_level
        self._id = id
        self._children = children
        self._class_counts = defaultdict(int)
        self._noisy_class_counts = None
        self._signal_to_noise = 0.0

    def add_child(self, child_node):
        self._children.append(child_node)

    def remove_child(self, child_node):
        value = child_node._split_value_from_parent
        if value not in [x._split_value_from_parent for x in self._children]:
            return 0
        else:
            new_children = [x for x in self._children if x._split_value_from_parent != value]
            self._children = new_children
            return 1


    def increment_class_count(self, class_value):
        self._class_counts[str(class_value)] += 1


    def make_noisy_class_counts(self, epsilon, class_values):
        ''' Add noise to the class counts of this leaf. Not performed on parents. '''
        if not self._noisy_class_counts and not self._children: # to make sure this code is only run once per leaf
            counts = {}
            for val in class_values:
                if val in self._class_counts:
                    counts[val] = max( 0, int(self._class_counts[val] + np.random.laplace(scale=float(1./epsilon))) )
                else: # original count was 0
                    counts[val] = max( 0, int(np.random.laplace(scale=float(1./epsilon))) )
            self._noisy_class_counts = counts
            self._signal_to_noise = (epsilon * 1.0 * sum([v for k,v in counts.items()])) / (math.sqrt(2*1.0) * len(counts)) # enS / (sqrt(2n) * num_classes)
            #print("self._signal_to_noise = {}".format(self._signal_to_noise))
            return 0
        else:
            return 1 # we're summing the redundant calls


    def add_childrens_class_counts(self, epsilon):
        ''' Add up the class counts of all leaf nodes that trace back to this parent node. '''
        if not self._noisy_class_counts and self._children:
            counts, num_leafs = self._find_leafs_and_count(self, defaultdict(int), 0)
            self._noisy_class_counts = counts
            self._signal_to_noise =  (epsilon * num_leafs * sum([v for k,v in counts.items()])) / (math.sqrt(2*num_leafs) * len(counts)) # enS / (sqrt(2n) * num_classes)
            return 0
        else:
            return 1 # we're summing the redundant calls
                    
    def _find_leafs_and_count(self, node, counts, num_leafs):
        if node._children:
            for child in node._children:
                counts, num_leafs = self._find_leafs_and_count(child, counts, num_leafs)
        else:
            num_leafs += 1
            for k,v in node._noisy_class_counts.items():
                counts[k] += v
        return counts, num_leafs


''' A toy example of the inputs and outputs of the SNR Differentially-Private Random Forest 
    Sometimes crashes. I haven't put the time in to debug it. Rerun the code to see if it stops crashing. '''
if __name__ == '__main__':
    data = [
            [1,'a','a','a','a','a'],
            [1,'a','a','b','b','b'],
            [1,'a','b','a','b','c'],
            [0,'a','b','b','a','d'],
            [0,'b','b','b','b','e'],
            [0,'b','b','a','b','f'],
            [1,'c','c','a','b','f'],
            [1,'c','c','a','b','f'],
            [1,'c','b','a','c','f'],
            [0,'c','b','c','c','f'],
            [0,'c','b','c','d','f'],
            [0,'c','b','c','d','f'],
            ]
    epsilon = 0.2
    our_diff_priv_forest = SNR_DP_Forest(data[2:], epsilon)

    num_trees = len(our_diff_priv_forest._trees)
    av_prunings = np.mean([x._prunings for x in our_diff_priv_forest._trees])
    av_tree_size = np.mean([x._id-x._prunings+1 for x in our_diff_priv_forest._trees])
    accuracy, leafs_not_most_confident, votes_requiring_average = our_diff_priv_forest.evaluate_accuracy_with_voting(data, class_index=0) # confidence count includes all trees (before voting)
    print("accuracy = {}".format(accuracy))