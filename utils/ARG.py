import numpy as np

# There is some problem with class node and class edge.
# (The attribute has error)
# But we only use ARG.nodes_vector and ARG.edges_matrix in this case.


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