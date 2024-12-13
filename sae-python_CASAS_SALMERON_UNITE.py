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


