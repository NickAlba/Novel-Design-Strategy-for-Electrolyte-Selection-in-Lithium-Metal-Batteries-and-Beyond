# Novel-Design-Strategy-for-Electrolyte-Selection-in-Lithium-Metal-Batteries-and-Beyond
The repository contains the source codes and the raw data used in the paper titled "Novel Design Strategy for Electrolyte Selection in Lithium Metal Batteries and Beyond".  
In particular, the repository provides the source code required to reproduce the computational analyses presented in the paper.
# Workflow
The analysis consists of three steps:
1. **Mechanical feasibility**  
   `scripts/01_Mechanical_feasibility/`  
   Compares the classification models and identifies the logistic
   regression model used in the subsequent analysis. The script also
   generates Figure 2a.
2.	**Regularization parameter selection**  
  `scripts/02_Lambda_selection/`  
  Determines the regularization parameter for the selected logistic
  regression model. The script generates Figure S2 in Supplementary Information.
3.	**Constrained optimization**  
`scripts/03_Constrained_optimization/`  
  Uses the selected logistic regression model, models ionic conductivity,
  and performs the final constrained optimization. The script generates
  Figure 2b.
  
Each folder contains the corresponding Python script and reference CSV
dataset.
# Dependencies
numpy  
pandas    
scikit-learn  
scipy  
pysr  
sympy  
matplotlib
