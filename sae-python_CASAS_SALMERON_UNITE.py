!pip3 install -U ucimlrepo

import ucimlrepo

from ucimlrepo import fetch_ucirepo 
  
# Installer le package ucimlrepo
#pip install ucimlrepo

# Importer les données  
from ucimlrepo import fetch_ucirepo 
  
# Récupérer l'ensemble de données
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# Les données (sous forme de dataframes pandas) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 


# metadata 
#print(statlog_german_credit_data.metadata) 
  
# variable information 
#print(statlog_german_credit_data.variables) 
print(statlog_german_credit_data.variables[['name','description']])


# Chargement des libraiies numpy, pands, matplolib, math et  random
import numpy as np
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
#from scipy.io import loadmat
import matplotlib.pylab as plt
import pandas as pd

import ucimlrepo

from ucimlrepo import fetch_ucirepo 
  
# Installer le package ucimlrepo
#pip install ucimlrepo

# Importer les données  
from ucimlrepo import fetch_ucirepo 
  
# Récupérer l'ensemble de données
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# Les données (sous forme de dataframes pandas) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 


# metadata 
#print(statlog_german_credit_data.metadata) 
  
# variable information 
#print(statlog_german_credit_data.variables) 
print(statlog_german_credit_data.variables[['name','description']])


# Chargement des libraiies numpy, pands, matplolib, math et  random
import numpy as np
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
#from scipy.io import loadmat
import matplotlib.pylab as plt
import pandas as pd


def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1d(X, epsilon=1e-10):
    A = np.array(X)
    
    n, m = A.shape
    
    # Transformation de A en une  matrice stochastique de dimension (n+m) * (n+m)
    Dr = np.diag(A.sum(axis=1))
    Dc = np.diag(A.sum(axis=0))
    
    Dc_1 = np.linalg.inv(Dc)
    Dr_1 = np.linalg.inv(Dr)
    
    col1 = np.concatenate([np.zeros((n,n)), np.dot(Dc_1 , A.T)])
    col2 = np.concatenate([np.dot(Dr_1 , A), np.zeros((m,m))])
    
    S = np.concatenate([col1, col2], axis=1)
    
    # initialisation du vecteur currentV
    x = randomUnitVector(n+m)
    lastV = None
    currentV = x
    
    lastE = np.linalg.norm(currentV)

    # Itérations 
    iterations = 0
    while True:
        iterations += 1
        lastV = np.array(currentV)
        currentV = np.dot(S, lastV)
        currentV = currentV / norm(currentV)
        
        last_u = lastV[list(range(0,n))]
        last_v = lastV[list(range(n,n+m))]
        
        current_u = currentV[list(range(0,n))]
        current_v = currentV[list(range(n,n+m))]
        
        e_u = np.linalg.norm(current_u - last_u)
        e_v = np.linalg.norm(current_v - last_v)
        
        currentE = e_u + e_v
        
        d = abs(currentE - lastE)
        lastE = currentE
        
        if d <= epsilon:
            print("converged in {} iterations!".format(iterations))

            #u = currentV[range(0,n)]
            #v = currentV[range(n,n+m)]
            
            return current_u, current_v


  X = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                 [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                 [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]])

columns=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]
index=["HighSchool", "AgricultCoop", "Railstation", "OneRoomSchool", "Veterinary", "NoDoctor", "NoWaterSupply",  "PoliceStation", "LandReallocation"]



#%%script false --no-raise-error
def reordonner(X, u, v):
    """
        renvoie les lignes et les colonnes de X réorganisées  par rapport au tri des vecteurs  u et v
        utiliser argsort() pour trier les vecteurs u et v
    """
    
    à completer le code ici...
 
    return X_reordonnee
