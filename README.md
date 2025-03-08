# AI-2-classifier
Development of the ML-based AI-2 QSIMs classifier
The codes for indentifying the potential AI-2 QSIMs.

1.To develop a general classifier for AI-2 QSIMs, we introduced the collected 215 reported compounds as positive samples, while 488 non-AI-2 QSIMs were employed as negative samples (Supplementary Table 2). 
2.The simplified molecular input line-entry system (SMILES) of both positive and negative compounds were converted into molecular graphs with topological information using RDKit (www.rdkit.org) and encoded with graph neural networks (GNN). Specifically, with the help from DeepChem70, we determined that the node feature vector includes one-hot encoding of the atom element, one-hot encoding of the degree of the atom in the molecule, number of directly-bonded neighbors (atoms), one-hot encoding of the total number of H bound to the atom, one-hot encoding of the number of implicit H bound to the atom, and whether the atom is aromatic. Then, an indirect binary graph with attribute nodes was constructed for the corresponding input SMILES strings of the positive and negative samples.
3.A fivefold cross-validation was performed to train the AI-2 QSIMs classifier. 
4.The performance of the classifier was evaluated based on the average area under the receiver operating characteristic curve (AUC), precision, accuracy, recall, and F1 score.
5.Codes in this file were used for the screening of the pythochemicals from the CMAUP database and the Phenol-Explorer database.
