## 3D GNN Github: https://github.com/divelab/DIG/tree/dig-stable/dig/threedgraph/method

some graph has unconnected components, causing nan values in angle calculations. this is dealt with, but need to verify whether those unconnected components make sense. 

the GNN model suffer from oversmoothing problem, less layers perform better 

try: layer: 1, 2
      n_output_layer (linear layer in ComENet): 2, 3 (this is new)
      hidden: 64, 128
      out: [512], [256, 512], [128, 512]


model overfit, training set can perform much better but the validation set is not improving 

final_*.pt, saved the best training set performance. 

hidden dim 512 layers 3, 3, outhidden 512 256 performs well 

hidden dim 768 does not work



1/16/24
 Using spherenet, edge features to do prediction. Only use the edges that connects H and C

 THe following DP is giving nan values for the algorithm. 
 '19600', '09140', '09139', '32257', '15523', '08951', '33227', '31566', '19601', '31508', '09412', '32275', '32702', '14673', '15501', '22433', '08953', '09303', '32258', '32255', '33234', '32646', '34267', '35202', '20947', '09522', '15525', '31658', '32394', '31633', '00573', '31579', '33519', '07629', '31918', '15527', '31036', '34263', '15526', '07558', '32520', '33582', '09313', '32261', '32272', '33884', '15502', '32185', '35149', '34268', '33226', '00867', '34269', '33495', '30777', '13906', '15114', '32564', '30926', '31615', '09430', '32369', '14809', '32640', '33225', '31640', '31120', '30927', '00571', '08952', '32237', '35226', '22434', '32274', '09523', '32222', '34238', '31655', '07627', '31795', '35147', '00380', '09304', '15366', '31523', '15113', '09278', '30813', '09992', '32565', '33302', '31176', '07628', '13907', '09296', '00570', '33237', '32420', '32597', '31932', '32266', '31201', '15707', '35598', '15115', '32422', '09214', '09300', '09264', '08526', '32269', '34704', '00381', '32635', '34705', '09855', '31430'


 still does not work



 1/20/24 
 use C alignment data. predict C shift first for each C node. then apply the gnn to 2dnmr data, label each C with shift, use this shift to predict the corresponding H shift

 Moved 2d nmr's 3d graph, as well as c alignment data's 3d graph to scratch 0

 --scratch0

----2dnmr_30k  
      ---graph_3d  
      ---nmr_1dcsv_30k 
      ---nmr_1dcsv_C_30k

----nmr_alignment
      ---graph3d