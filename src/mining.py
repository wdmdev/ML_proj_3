from data_cleaner import create_dataset
import numpy as np
from apyori import apriori

data, pred = create_dataset()

data = np.asarray(data)

X = np.zeros(data.shape)

means = np.mean(data,axis=0)

for i, obs in enumerate(data):
    X[i] = obs > means

label = [
  "PV",
  "TA",
  "GD",
  "VE",
  "TE",
  "PS",
  "NW",
  "JA"
]
print(X)
def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T


T = mat2transactions(X,label)

rules = apriori(T, min_support = 0.25, min_confidence = 0.6)

def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:  
            conf = o.confidence
            supp = r.support
            x = ", ".join( list( o.items_base ) )
            y = ", ".join( list( o.items_add ) )
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append( (x,y) )
    return frules
# Print rules found in the courses file.
print(print_apriori_rules(rules))


#association_results = list(association_rules)
"""
def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:        
            conf = o.confidence
            supp = r.support
            x = ", ".join( list( o.items_base ) )
            y = ", ".join( list( o.items_add ) )
            #print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append( (x,y) )
    return frules

print(print_apriori_rules(rules))
"""