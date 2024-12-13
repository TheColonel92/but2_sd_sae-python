import ucimlrepo

from ucimlrepo import fetch_ucirepo 
  
# Récupérer l'ensemble de données
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# Les données (sous forme de dataframes pandas) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 

print(statlog_german_credit_data.variables[['name','description']])
