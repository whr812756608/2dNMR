import torch
from torch_geometric.data import DataLoader as loader_2dnmr
from torch.utils.data import Dataset
import pandas as pd
from Comenet_NMR import ComENet
import time
import numpy as np
import pickle
import os
from Comenet_NMR_multitask import ComENet
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from matching_code.ARG import ARG
from sklearn.preprocessing import normalize


class graph_nmr_data_2d(Dataset):
    '''Returns the index of non-zero values on y-axis and the corresponding x-axis'''
    def __init__(self, csv_file, graph_path, nmr_path, x='C'):
        df = pd.read_csv(csv_file)
        self.file_list = df['File_name'].to_list()
        self.solvent_class = df['solvent_class'].to_list()
        # filter out Test_*.csv
        # self.file_list = [x for x in self.file_list if not x.startswith('Test_')]

        self.nmr_path = nmr_path
        self.graph_path = graph_path
        self.x = x
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, item):
        filename = self.file_list[item].split('.')[0]
        solvent_class = torch.tensor(self.solvent_class[item])

        graph_file = os.path.join(self.graph_path, filename + '.pickle')
        graph_data = pickle.load(open(graph_file, 'rb'))
        graph_data.x = graph_data.x.float()
        
        graph_data.has_c = True
        graph_data.has_h = True
        graph_data.solvent_class = solvent_class

        # use original csv files so that the C nodes are not duplicated
        nmr_file = os.path.join(self.nmr_path, filename + '.csv')
        nmr_data = pd.read_csv(nmr_file)

        # Forward fill the 'No.' column to fill empty values with the previous row's value
        if 'No.' in nmr_data.columns:
            nmr_data['No.'].fillna(method='ffill', inplace=True)
            # Group by both 'No.' and '13C' columns and create a list of '1H' values
            grouped = nmr_data.groupby(['No.', '13C'])['1H'].apply(list).reset_index()
        else:
            grouped = nmr_data.groupby('13C')['1H'].apply(list).reset_index()
        c_list = grouped['13C'].tolist()
        h_list = grouped['1H'].apply(lambda x: x if len(x) > 1 else [x[0], x[0]]).tolist()
        c_peaks = torch.tensor(c_list).float().squeeze()/200
        try:
            h_peaks = torch.tensor(h_list).float()/10
        except Exception as e:
            print(f"Error while converting to tensor for filename: {filename}")
            print(e)
            h_peaks = torch.tensor([])

        return graph_data, c_peaks, h_peaks, filename
    
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

    
hc = 512
chc=[128, 64]
hhc=[128, 64]
n = 3
n_out = 3
dropout = 0.4
initial_match = False


if initial_match:
    model = ComENet(in_embed_size=3, c_out_channels=1, h_out_channels=1, agg_method='sum',\
        hidden_channels=hc, c_out_hidden=chc, h_out_hidden=hhc, num_layers=n, num_output_layers=n_out, dropout=dropout)
    msg = model.load_state_dict(torch.load(
        'gnn3d_multi_align_sum_hiddendim_%d_nlayers_%d_noutlayers_%d_couthidden_%s_houthidden_%s.pt'\
        %(hc, n, n_out, ''.join(str(i) for i in chc), ''.join(str(i) for i in hhc)), map_location='cpu'))
    print('loading initial model ', msg)
else:
    model = ComENet(in_embed_size=3, c_out_channels=1, h_out_channels=2, agg_method='sum',\
        hidden_channels=hc, c_out_hidden=chc, h_out_hidden=hhc, num_layers=n, num_output_layers=n_out, dropout=dropout)
    msg = model.load_state_dict(torch.load(
        'gnn3d_nofreeze_solvent_2dch_sum_hiddendim_%d_nlayers_%d_noutlayers_%d_couthidden_%s_houthidden_%s.pt'\
        %(hc, n, n_out, ''.join(str(i) for i in chc), ''.join(str(i) for i in hhc)), map_location='cpu'))
    print('loading finetuned model ', msg)

model.eval()

graph_path_2dnmr = '/scratch0/yunruili/2dnmr_30k/graph_3d/'
csv_file_2dnmr = 'nmr_smile_solventclass_filtered2_3dgnn.csv'
nmr_path_2dnmr = '/scratch0/yunruili/2dnmr_30k/csv_30k/'
save_dir = '/scratch0/yunruili/2dnmr_30k/nmr_2dcsv_chmatched_2nd_2'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

dataset_2dnmr = graph_nmr_data_2d(csv_file_2dnmr, graph_path_2dnmr, nmr_path_2dnmr)
dataloader_2dnmr = loader_2dnmr(dataset_2dnmr, shuffle=True, batch_size=1)

list_files = []
for i, data in enumerate(dataloader_2dnmr):
    graph, cnmr, hnmr, filename = data
    
    c_nodes = (graph.x[:,0]==5).nonzero(as_tuple=True)[0]
    h_nodes = (graph.x[:, 0] == 0).nonzero(as_tuple=True)[0] 

    c_shifts, h_shifts = model(graph)

    ##### calculate the indices of C node connected to H
    # Initialize a list to store C nodes connected to H
    c_nodes_connected_to_h = []
    # Check each C node for connection to any H node
    for c_node in c_nodes:
        # Get indices of edges involving the C node
        edges_of_c = (graph.edge_index[0] == c_node) | (graph.edge_index[1] == c_node)

        # Get all nodes that are connected to this C node
        connected_nodes = torch.cat((graph.edge_index[0][edges_of_c], graph.edge_index[1][edges_of_c])).unique()

        # Check if any of these connected nodes are H nodes
        if any(node in h_nodes for node in connected_nodes):
            c_nodes_connected_to_h.append(c_node.item())
    
    # Convert to a tensor
    c_nodes_connected_to_h = torch.tensor(c_nodes_connected_to_h)

    c_shifts = c_shifts.detach().numpy()
    h_shifts = h_shifts.detach().numpy()
    cnmr = cnmr.squeeze().detach().numpy()
    hnmr = hnmr.squeeze().detach().numpy()

    # print(c_shifts.shape, h_shifts.shape)
    # print(cnmr.shape, hnmr.shape)

    c_index = [i for i, x in enumerate(c_nodes) if x in c_nodes_connected_to_h]
    c_shifts = c_shifts[c_index]

    # print(filename)
    # print(len(c_shifts), len(h_shifts))
    # print(len(cnmr), len(hnmr))

    # make c-h pairs for ground truth and prediction
    hnmr2 = np.mean(hnmr, axis=1)
    if not initial_match:
        h_shifts = np.mean(h_shifts, axis=1, keepdims=True)
    pred = np.concatenate([c_shifts, h_shifts], axis=1)
    gt = np.stack([cnmr, hnmr2], axis=1)

    try:
        if len(cnmr) == len(c_shifts):
            rslt = match_samelen(pred, gt)
        else:
            rslt = match_difflen(pred, gt)

        cnmr_expanded = np.array([[cnmr[i]] for i in rslt[:,1]])
        hnmr_expanded = np.array([hnmr[i] for i in rslt[:, 1]])

        # save as pickle
        c_name = os.path.join(save_dir, '%s_c.pickle'%filename)
        h_name = os.path.join(save_dir, '%s_h.pickle'%filename)
        
        # Writing to a pickle file
        with open(c_name, 'wb') as file:
            pickle.dump(cnmr_expanded, file)
        with open(h_name, 'wb') as file:
            pickle.dump(hnmr_expanded, file)
    except Exception as e:
        print(filename)
        list_files.append(filename)
        print(e)