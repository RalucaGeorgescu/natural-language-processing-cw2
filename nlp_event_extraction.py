import numpy as np
import networkx as nx
from collections import defaultdict


from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# converts labels into integers, and vice versa, needed by scikit-learn.
label_encoder = LabelEncoder()

# encodes feature dictionaries as numpy vectors, needed by scikit-learn.
vectorizer = DictVectorizer()

def is_argument_candidate(event, index):
    for x, y in event.argument_candidate_spans:
        if index >= x and index < y:
            return True
    return False 

def distance_to_argument(event, index):
    abs_minimum = len(event.sent.tokens)
    minimum = len(event.sent.tokens)
    for arg in event.argument_candidate_spans:
        if abs(arg[0]-index) <= abs_minimum:
            abs_minimum = abs(index-arg[0])
            minimum = arg[0] - index
        if abs(arg[1]-index) <= abs_minimum:
            abs_minimum = abs(index-arg[0])
            minimum = arg[1] - index
    return minimum

def distance_to_protein(event, index):
    abs_minimum = len(event.sent.tokens)
    minimum = len(event.sent.tokens)
    for mention in event.sent.mentions:
        if abs(mention['begin']-index) <= abs_minimum:
            abs_minimum = abs(index-mention['begin'])
            minimum = mention['begin'] - index
        if abs(mention['end']-index) <= abs_minimum:
            abs_minimum = abs(index-mention['end'])
            minimum = mention['end'] - index
    return minimum

def event_feat(event):
    """
    This feature function returns a dictionary representation of the event candidate. You can improve the model 
    by improving this feature function.
    Args:
        event: the `EventCandidate` object to produce a feature dictionary for.
    Returns:
        a dictionary with feature keys/indices mapped to feature counts.
    """
    result = defaultdict(float)
    index = event.trigger_index
    result['trigger_word=' + event.sent.tokens[index]['word']] += 1.0   
    result['trigger_stem=' + event.sent.tokens[index]['stem']] += 1.0
    result['trigger_pos=' + event.sent.tokens[index]['pos']] += 1.0
    result['trigger_protein'] = 1.0 if event.sent.is_protein[index] else 0.0
    result['trigger_suffix3=' + event.sent.tokens[index]['word'][-3:]] += 1.0
    result['trigger_suffix2=' + event.sent.tokens[index]['word'][-2:]] += 1.0
    if index > 0:
        result['preceeding_pos=' + event.sent.tokens[index-1]['pos']] += 1.0
    if index > 1:
        result['pre_preceeding_pos=' + event.sent.tokens[index-2]['pos']] += 1.0
    if index < len(event_candidate.sent.tokens)-1:
        result['next_pos=' + event.sent.tokens[index+1]['pos']] += 1.0      
    if index > 4:
        prev_5 = ""
        prev_5_pos = ""
        for i in range(index - 5, index):
            prev_5 += event.sent.tokens[i]['word'] + "->"
            prev_5_pos += event.sent.tokens[i]['pos'] + "->"
        result['prev_5: ' + prev_5] += 1.0        
        result['prev_5_pos: ' + prev_5_pos] += 1.0
    if index < len(event_candidate.sent.tokens) - 6:
        next_2 = ""
        for i in range(index+1, index+2):
            next_2 += event.sent.tokens[i]['word'] + "->"
        result['next_2: ' + next_2] += 1.0    
    
    # Only if there are protein mentions in the event candidate sentence
    if not event.sent.mentions == False:
        result['distance_protein='] = distance_to_protein(event, index)
    
        nearest_protein_index = index + distance_to_protein(event, index)
        nearest_protein_words = ""
        nearest_protein_stems = ""
        begin_protein_index = -1

        for mention in event.sent.mentions:
            if mention['begin'] == nearest_protein_index or mention['end'] == nearest_protein_index:
                begin_protein_index = mention['begin']
                for i in range(mention['begin'], mention['end']):
                    nearest_protein_words += event.sent.tokens[i]['word'] + " "
                    nearest_protein_stems += event.sent.tokens[i]['stem'] + " "
                    
        graph_list = []

        for dep in event.sent.dependencies:
            graph_list.append((dep['head'], dep['mod']))

        graph = nx.Graph()
        for x,y in graph_list:
            graph.add_edge(x,y)

        if index in graph and begin_protein_index in graph and nx.has_path(graph, index, begin_protein_index): 
            shortest_path = nx.shortest_path(graph, index, begin_protein_index)
            result['dependency_path_length'] = len(shortest_path)
            
            path=""
            for i in range(0, len(shortest_path)-1):
                head = shortest_path[i]
                mod = shortest_path[i+1]
                for dep in event.sent.dependencies:
                    if (dep['mod'] == mod and dep['head'] == head) or (dep['mod'] == head and dep['head'] == mod):
                        path += dep['label'] + "->"
            result['dependency_path: ' + path] += 1.0
            
        result['nearest_protein:' + nearest_protein_words] = 1.0
        result['nearest_protein_stem:' + nearest_protein_stems] = 1.0 
    
    if len(event.argument_candidate_spans) != 0:
        result['distance_argument'] = distance_to_argument(event, index)
    
    for child,label in event.sent.children[index]:
        result['child_obj: ' + event.sent.tokens[child]['word']] = 1.0 if label == 'dobj' or label == 'iobj' or label == 'pobj' else 0.0

    for child,label in event.sent.children[index]:
        result["child: " + label + "->"+ event.sent.tokens[child]['word'] + "->" + event.sent.tokens[child]['stem'] + "->" + str(event.sent.is_protein[child]) + "->" + str(is_argument_candidate(event, child))+ "->"+ event.sent.tokens[child]['pos']] += 1.0 
    for parent,label in event.sent.parents[index]:
        result["parent: " + label + "->"+ event.sent.tokens[parent]['word'] + "->"+ event.sent.tokens[parent]['stem'] + "->" + str(event.sent.is_protein[parent]) + "->" + str(is_argument_candidate(event, parent))+ "->"+ event.sent.tokens[parent]['pos']] += 1.0
        result["parent_pos:" + event.sent.tokens[parent]['pos']] += 1.0
        result["parent_label:" + label] += 1.0
    
    result['has_children'] = 0.0 if len(event.sent.children[index]) == 0 else 1.0
    result['has_parents'] = 0.0 if len(event.sent.parents[index]) == 0 else 1.0
            
    return result

# Separate training and dev sets from the inital training set
event_train_split = event_train[:len(event_train)//3 * 2]
event_dev_split = event_train[len(event_train)//3 * 2:]

# We convert the event candidates and their labels into vectors and integers, respectively.
train_event_x = vectorizer.fit_transform([event_feat(x) for x,_ in event_train])
train_event_y = label_encoder.fit_transform([y for _,y in event_train])

lr = LogisticRegression(C=10, class_weight = 'balanced', verbose = 1)
lr = lr.fit(train_event_x, train_event_y)

def predict_event_labels(event_candidates):
    """
    This function receives a list of `bio.EventCandidate` objects and predicts their labels. 
    It is currently implemented using scikit-learn, but you are free to replace it with any other
    implementation as long as you fulfil its contract.
    Args:
        event_candidates: A list of `EventCandidate` objects to label.
    Returns:
        a list of event labels, where the i-th label belongs to the i-th event candidate in the input.
    """
    event_x = vectorizer.transform([event_feat(e) for e in event_candidates])
    event_y = label_encoder.inverse_transform(lr.predict(event_x))
    return event_y