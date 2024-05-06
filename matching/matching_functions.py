import torch
from torch_geometric.data import DataLoader as loader_2dnmr
from torch.utils.data import Dataset
import pandas as pd
import time
import numpy as np
import pickle
import os
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

class node:

    '''
    Node is a class representing the point in a graph
    Node will have edges(also class) connected to it 
    '''

    def __init__(self, id, ARG=None):
        self.ID = id
        self.ARG = ARG
    
    def has_atrs(self): # check if the node has attribute
        if self.ARG is None:
            return False
        else:
            return True
    
    def get_atrs(self):
        return self.ARG.nodes_vector[self.ID]  # Define in class AGC
        
    def num_atrs(self):
        return len(self.get_atrs())       
        
        
class edge:
    '''
    Edge is the connection between nodes
    It will have assigned weight and two end points (nodes)
    '''
    def __init__(self, node1, node2, AGR=None):
        self.node1 = node1
        self.node2 = node2
        self.AGR = AGR
    def has_atrs(self):
        if self.ARG is None:
            return False
        else:
            return True
    
    def get_atrs(self):
        return self.ARG.edges_matrix[self.node1,self.node2]
    
    def num_atrs(self):
        return len(self.get_atrs())  
    
class ARG:
    '''
    Attributed Relational Graphs represents a graph data structure with a list of node.
    M is the edge weight matrix and V is the node attributes.
    
    '''
    def __init__(self, M, V):
        
        # Check if M is square
        assert (M.shape == M.transpose().shape), "Input edge weight matrix is not square."
        # Check if numbers of nodes from M and from V is matched.
        assert (len(M) == len(V)), "Input sizes of edge weight matrix and of nodes attributes do not match."
        
        # Use dictionary structure to store nodes and edges.
        self.num_nodes = len(M)
        self.nodes = {}
        self.edges = {}
        self.nodes_vector = V.reshape([self.num_nodes,-1])
        # For nodes
        for id in range(self.num_nodes):
            self.nodes[id] = node(id)
            #self.nodes_vector[id] = V[id]
        
        # For edges
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.edges[(i,j)] = edge(i,j)                
        self.edges_matrix = M

def match_samelen(pred, gt):
    # Calculate the cost (distance) matrix
    cost_matrix = cdist(pred, gt, metric='euclidean')

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Output the matched pairs
    matched_pairs = np.array(list(zip(row_ind, col_ind)))
    return matched_pairs

def compatibility(atr1, atr2):
    #Consider the order to return 0 or inf
    
    if np.isnan(atr1).any() or np.isnan(atr1).any():
        return 0
    if (atr1 == float('Inf')).any() or (atr2 == float('Inf')).any():
        return float('Inf')
    if len(atr1) != len(atr2):
        return 0
    if (atr1 == 0).all() or (atr2 == 0).all():
        return 0
    
    
    dim = len(atr1)
    score = 1-((atr1-atr2)**2).sum()
    #score = atr1 * atr2
    return score

def pre_compute_compatibility(ARG1, ARG2, alpha=1, stochastic=0, node_binary=True, edge_binary=True, dist_mask=None):
    '''
    Compute the best matching with two ARGs.
    
    '''
    
    # Size of the real match-in matrix
    A = ARG1.num_nodes
    I = ARG2.num_nodes
    real_size = [A, I] # ???
    augment_size = [A+1, I+1] # size of the matrix with slacks
    
    #### compute c_aibj ####
    # nil node compatibility percentage
    prct = 10
    
    ## pre-calculate the node compatibility
    C_n = np.zeros(augment_size)
    
    if node_binary:
        C_n[:A,:I] = cdist(ARG1.nodes_vector, ARG2.nodes_vector, compatibility_binary)
    else:
        C_n[:A,:I] = cdist(ARG1.nodes_vector, ARG2.nodes_vector, compatibility)
    
    # Add score to slacks
    C_n[A,:-1] =  np.percentile(C_n[:A,:I],prct,0)
    C_n[:-1,I] =  np.percentile(C_n[:A,:I],prct,1)
    C_n[A,I] = 0 

    # times the alpha weight
    C_n = alpha*C_n
    
#     print(C_n)
    
    if dist_mask is not None:
        C_n[:A,:I] = np.multiply(C_n[:A,:I], dist_mask)
    
    return C_n

def graph_matching(C_n, ARG1, ARG2, beta_0=0.1, beta_f=20, beta_r=1.025, 
                   I_0=20, I_1=200, e_B=0.1, e_C=0.01, fixed_match = None):  # C_e, 
    ### fixed match is a matrix (A*I) for the pre-determined matching pairs 
    ##  We first do not consider the stochastic.
    # set up the soft assignment matrix
    
    A = C_n.shape[0] - 1
    I = C_n.shape[1] - 1 
    m_Head = np.random.rand(A+1, I+1) # Not an assignment matrix. (Normalized??)
    m_Head[-1,-1] = 0
    
    ### zero the nodes that already matched 
    if fixed_match is not None:
        print('fixed some points')
        C_n = np.multiply(C_n, fixed_match)

    # Initialization for parameters

    ## beta is the penalty parameters
    # includes beta_0, beta_f, beta_r

    ## I controls the maximum iteration for each round
    # includes I_0 and I_1

    ## e controls the range
    # includes e_B and e_C

    # begin matching
    beta = beta_0

    stochastic = False ### we first do not consider this case
    
    # the indexes of the non-zero elements in C_n
    idx1 = np.unique(C_n.nonzero()[0])
    idx2 = np.unique(C_n.nonzero()[1])  # not used, only check number


    while beta < beta_f:

        ## Round-B
        #check if converges
        converge_B = False
        I_B = 0
        while (not converge_B) and I_B <= I_0: # Do B until B is converge or iteration exceeds
            if stochastic:
                m_Head = m_Head ### + ???           

#             print('I_B', m_Head[0])
            old_B = m_Head # the old matrix
            I_B += 1 

            # Build the partial derivative matrix Q
            Q = np.zeros([A+1, I+1])

            # Node attribute
            Q = Q + C_n 
#             print(Q)

            # Update m_Head
            m_Head = np.exp(beta*Q) 
            m_Head[-1, -1] = 0
            
#             print(m_Head)
            
            converge_C = False
            I_C = 0
            while (not converge_C) and I_C <= I_1: # Do C until C is converge or iteration exceeds
                I_C += 1
                old_C = m_Head
                
#                 print(m_Head[0])

                # Begin alternative normalization. 
                # Do not consider the row or column of slacks
                # by column
                m_Head = normalize(m_Head, norm='l2',axis=0)*normalize(m_Head, norm='l2',axis=0)
                # By row
                m_Head = normalize(m_Head, norm='l2',axis=1)*normalize(m_Head, norm='l2',axis=1)
                
#                 print('I_C', m_Head[0])

                # print(sum(m_Head))
                # update converge_C
                converge_C = abs(sum(sum(m_Head-old_C))) < e_C

            # update converge_B
            converge_B = abs(sum(sum(m_Head[:A,:I]-old_B[:A,:I]))) < e_B
        # update beta
        beta *= beta_r

    match_matrix = heuristic(m_Head, A, I)
    #match_matrix = m_Head
    return match_matrix

def heuristic(M, A, I):
    '''
    Make a soft assignment matrix to a permutation matrix. 
    Due to some rules.
    We just set the maximum element in each column to 1 and 
    all others to 0.
    This heuristic will always return a permutation matrix 
    from a row dominant doubly stochastic matrix.
    '''
    M = normalize(M, norm='l2',axis=1)*normalize(M, norm='l2',axis=1)
    for i in range(A+1):
        index = np.argmax(M[i,:]) # Get the maximum index of each row
        M[i,:] = 0
#         if index != I-1:
#             M[:,index] = 0
#         ###
#         else:
#             print(i, I-1)
        M[i,index] = 1
    M = M[:A,:I]
    return M

def match_difflen(pred, gt):
    edges_pred = np.zeros((len(pred), len(pred)))
    edges_gt = np.zeros((len(gt), len(gt)))

    dist_matrix = cdist(pred, gt)
    ARG1 = ARG(edges_pred, pred)
    ARG2 = ARG(edges_gt, gt)
    start_time = time.time()
    C_n = pre_compute_compatibility( ARG1, ARG2, alpha=1, stochastic=0,node_binary=False)
    # print("--- Calculate C_n,  %s hours ---" % ((time.time() - start_time)/3600))
    start_time = time.time()
    match_matrix = graph_matching(C_n=C_n, ARG1 = ARG1, ARG2 = ARG2, 
                                beta_0=0.1, beta_f=100, beta_r=1.01, 
                                I_0=200, I_1=200, e_B=0.00005, e_C=0.00005
                                )


    g1, g2 = match_matrix.nonzero()
    rslt = np.full([match_matrix.shape[0], 2], -1) # need to maintain record if one node is graph 1 is not matched

    rslt[:, 0] = np.arange(0, match_matrix.shape[0])


    for i in range(len(g1)):
        rslt[g1[i], 1] = g2[i] 
    
    # Find rows in array1 where the second column is -1
    rows_to_match = np.where(rslt[:, 1] == -1)[0]
    # For each of these rows, find the index of the minimum value in array2 and assign it
    for row in rows_to_match:
        rslt[row, 1] = np.argmin(dist_matrix[row])
    return rslt