import numpy as np
import random , math
import pandas as pd
from scipy . optimize import minimize 
import matplotlib . pyplot as plt
from sklearn.preprocessing import StandardScaler



# parameters is a datapoint di = xi = (x1i,x2i,...,xni)
# linear kernel
def linear_kernel(di,dj):
    ker = np.dot(di,dj)
    return ker

# Polynomial kernel
p = 3
def poly_kernel(di,dj):
    ker = (np.dot(di,dj) + 1)**p
    return ker

# Radial kernel
sigma = 0.5
def rad_kernel(di,dj):
    sq_dist = np.linalg.norm(di-dj)**2
    ker = math.exp(-sq_dist/(2*sigma**2))
    return ker

ker = rad_kernel   # Choosing Kernel 
C = None     # Choosing slack coeff, C = None if no slack


# Importing data as numpy arrays
from preprocessing import X_df,y_df
from sklearn.model_selection import train_test_split
# classA = X_df[y_df == 1].to_numpy()
# classA = classA[:,2:4]

# classB = X_df[y_df == 0].to_numpy()
# classB = classB[:,2:4]

X = X_df.to_numpy()
y = y_df.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# X_train = X_train[:,2:4]
N = X_train.shape[0]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# classA = scaler.transform(classA)
# classB = scaler.transform(classB)



# Construct P matrix including the factor 0.5
P = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        P[i][j] = 0.5*y_train[i]*y_train[j]*ker(X_train[i],X_train[j])
    
# Define objective function alphaT*P*alpha-the sum of alpha
def objective(alpha):
    res = np.dot(alpha,np.dot(P,alpha)) - np.sum(alpha)
    return res


# Define zerofunc sum of alphai*yi 
def zerofun(alpha):
    res = np.dot(alpha,y_train)
    return res


alpha0 = np.zeros(N)                # Starting value for alpha
B = [(0,C) for b in range(N)]       # Conditions for alphas, C = None => no slack
XC = {'type':'eq', 'fun':zerofun}   # Equality constraint 

ret = minimize( objective , alpha0 ,
                bounds=B, constraints=XC )
alpha = ret['x']
success = ret['success']
print('Found solution = {}'.format(success))
tol = 1e-5      # Tolerance for floating point error
# Separate and store non negative alphas with corresponing input and target
filtered_data = [ [value, X_train[idx], y_train[idx]] for idx, value in enumerate(alpha) if value > tol]

# Find marginal points and select one of them (no condition of which)
# marginal_point is a list of alpha with corresponing input and target

if C == None:       # If no slack
    marginal_points = [point for point in filtered_data]
    margin_point = marginal_points[0]
    
else:               # If slack one need to choose alpha < C
    marginal_points = [point for point in filtered_data if point[0] < C-tol ] # Added tolerance for float error
    margin_point = marginal_points[0]         
print('Marginal points: {}'.format(len(marginal_points)))

# Caluclate the sum of alphai*yi*K(s,xi)
def sum_sv(data, s):
    S = 0
    for point in data:
        alph = point[0]
        x = point[1]
        y = point[2]
        S += alph*y*ker(s,x)
    return S
