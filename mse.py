import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import gurobipy as gp
from gurobipy import GRB


filename = "data.csv"
df = pd.read_csv(filename, sep="\t")

# Groupping the dicts of groups, garments, and SKUs
groups = df.groupby("group")[["group", "garment", "size", "quantity"]].apply(lambda g: g.values.tolist()).to_dict()
garments = df.groupby(["group", "garment"])[["group", "garment", "size", "quantity"]].apply(lambda g: g.values.tolist()).to_dict()
skus = df.groupby(["group", "garment", "size"])[["group", "garment", "size", "quantity"]].apply(lambda g: g.values.tolist()).to_dict()

# Listing in groups, garments, and SKUs
list_groups = list(groups.keys())       
list_garments = list(garments.keys())
list_skus = list(skus.keys())

# Getting the garment of SKU(i)
def getGarment(i):
    if i > len(list_skus):
        return -1
    else:
        for g in list_garments:
            if set(g).issubset(list_skus[i]):
                return list_garments.index(g)
            
# Getting the group of SKU(i)
def getGroup(i):
    if i > len(list_skus):
        return -1
    else:
        for g in list_groups:
            if set(g).issubset(list_skus[i]):
                return list_groups.index(g)

# Getting quantity of SKU(i)
def getQuantity(i):
    if i > len(list_skus):
        return -1
    else:
        return skus.get(list_skus[i])[0][3]
    
def getNumberOfSKU(g):
    sum = 0
    for sku in list_skus:
        if set(list_groups[g]).issubset(sku):
            sum += 1
    return sum        

# Callback functions
def data_cb(model, where):
    if where == gp.GRB.Callback.MIP:
        cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
        gap = (abs(cur_bd - cur_obj)/abs(cur_obj))*100
        
        # Change in obj value or bound?
        if model._obj != cur_obj or model._bd != cur_bd:
            model._obj = cur_obj
            model._bd = cur_bd
            model._gap = gap
            model._data.append([time.time() - model._start, cur_obj, cur_bd, gap])

# Plotting the gap from callback
def plotGap(data):
        dfResult = pd.DataFrame(data, columns=['time', 'cur_obj','cur_bd','gap'])
        dfResult = dfResult.drop(dfResult[dfResult.cur_obj >= 100000000].index)
        
        fig, axes = plt.subplots()
        
        axes.set_xlabel('time')
        axes.set_ylabel('value')
        axes.set_xlim(dfResult['time'].values.min(), dfResult['time'].values.max())
        axes.set_ylim(0, dfResult['cur_obj'].values.max() * 1.1)
        line1, = axes.plot(dfResult['time'].values, dfResult['cur_obj'].values, color = 'navy', label='Current ObjValue')    
        line2, = axes.plot(dfResult['time'].values, dfResult['cur_bd'].values, color = 'blue', label='Current DB')    
        plt.fill_between(dfResult['time'].values, dfResult['cur_obj'].values, dfResult['cur_bd'].values, lw=0, color='lightsteelblue')
        
        axes2 = axes.twinx()
        axes2.set_ylabel('%gap')
        axes2.set_ylim(0, 100)
        line3, = axes2.plot(dfResult['time'].values, dfResult['gap'].values, color = 'red', label='Current Gap')
        axes.legend(handles=[line1, line2, line3], bbox_to_anchor=(0.5, 1.1), frameon=False, loc='upper center', ncol=3)
        
        plt.show()



### Main model
T = len(list_skus)                                          # Maximum number of squences = number of total different SKUs
I = len(list_skus)                                          # Number of total different SKUs
R = len(list_garments)                                      # Number of total different garments
G = len(list_groups)                                        # Number of total different groups
K = 3                                                       # Number of picking locations
l = 4                                                       # Number of conveyor lanes in each picking location
C = K * l                                                   # Maximum capacity (different SKUs) for a single sequence (here 12)
totalQuantile = sum(getQuantity(i) for i in range(I))       # Total quantity of all SKUs in a batch

model = gp.Model()

### Varibales used in model

xIndex = [(i, t) for i in range(I) for t in range(T)]
yIndex = [(g, t) for g in range(G) for t in range(T)]
zIndex = [(r, t) for r in range(R) for t in range(T)]
eIndex = [(t) for t in range(T)]

## Starting variables
x = model.addVars(xIndex, vtype=GRB.BINARY, name = "x")
y = model.addVars(yIndex, vtype=GRB.BINARY, name = "y")
z = model.addVars(zIndex, vtype=GRB.BINARY, name = "z")

## Helping variables for modelling
n = model.addVar(lb = int(I/(K*l)), ub = I, vtype=GRB.INTEGER, name = "n")      # Real number of sequences
n_inv = model.addVar(vtype=GRB.CONTINUOUS, name = "n_inv")                      # Inverse of number of sequences
n_temp = model.addVars(xIndex, vtype=GRB.INTEGER, name = "x")                   # Helping variable to calculate n
e = model.addVars(eIndex, lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name = "e") # Errors between real and expected workload of sequence t
e_squared = model.addVars(eIndex, vtype=GRB.CONTINUOUS, name = "e_squared")     # Squared errors between real and expected workload of sequence t
sse = model.addVar(vtype=GRB.CONTINUOUS, name = "sse")                          # Sum of squared deviation = sum of squared errors 
f = model.addVars(eIndex, vtype=GRB.INTEGER, name = "f")                        # There is a sequence t or not
### Objective function
model.setParam("NonConvex", 2)
model.setObjective(n_inv * sse, GRB.MINIMIZE)
model.update()

### Constraints
## Constraints in starting model (.doc file)
for i in range(I):
    model.addConstr(sum(x[i,t] for t in range(T)) == 1, name = "C1")
for t in range(T):
    model.addConstr(sum(x[i,t] for i in range(I)) <= C, name = "C2")
    model.addConstr(sum(y[g,t] for g in range(G)) <= 1, name = "C3")
    for i in range(I):
        model.addConstr(x[i,t] <= y[getGroup(i),t], name = "C4")
        model.addConstr(x[i,t] == z[getGarment(i),t], name = "C6")
for r in range(R):        
    model.addConstr(sum(z[r,t] for t in range(T)) == 1, name = "C5")

## Additional constraints

# Consecutive sequence number and number of sequences strarts from 1
for t1 in range(T):
    for t2 in range(T):
        if t1 > t2:
            model.addConstr(sum(y[g,t1] for g in range(G)) <= sum(y[g,t2] for g in range(G)), name = "C10")

# Consecutive sequences of a group
for g in range(G):
    for t1 in range(T):
        for t2 in range(T):
            if t1 > t2:
                model.addConstr(t1*y[g, t1] - t2*y[g,t2] <= T*(2 - y[g,t1] - y[g,t2]) + getNumberOfSKU(g)//C, name = "C11")
    
# Dertermining real number of sequences
for i in range(I):
    for t in range(T):
        model.addConstr(n_temp[i,t] == x[i,t]*t, name = "C12")
model.addConstr(n == gp.max_(n_temp), name = "C13")

# Helping devinding by a variable in objective function (1/n)        
model.addConstr(n*n_inv == 1, name = "C14")

# Helping multipling more than 2 variables (max. 2 with gurobi) in objective function
for t in range(T):
    model.addConstr(e[t] == totalQuantile*n_inv - sum(x[i,t]*getQuantity(i) for i in range(I)), name = "C15")
    model.addConstr(e_squared[t] == e[t]**2, name = "C16")    
    model.addConstr(f[t] == sum(x[i,t] for i in range(I)), name = "C17")
model.addConstr(sse == sum(f[t] * e_squared[t] for t in range(T)), name = "C18")

### Setting parameters
model._obj = None
model._bd = None
model._gap = None
model._data = []
model._start = time.time()

model.Params.TimeLimit = 3*60*60
model.update()

model.optimize(callback=data_cb)
model.write("mse.sol")
# model.computeIIS()
# model.write("model.ilp")


### Writting result 
xResult = pd.DataFrame(x.keys(), columns=["i","t"])
xResult["value"]=model.getAttr("X", x).values()
df["sequence"] = xResult.loc[xResult["value"] > 0.9, "t"].values + 1
print(df)
df.to_csv("mse.csv", encoding='utf-8')
plotGap(model._data)

