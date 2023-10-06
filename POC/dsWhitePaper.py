from scipy import sparse
from scipy.stats import uniform
from scipy.sparse import diags
import numpy as np
from functools import wraps
import time

####################################################################
# 3.1 - Simple Example
####################################################################

# 3.1.1 Price Matrix P
rawPrice = [100, 2]
P = np.array(rawPrice).reshape(len(rawPrice), 1)
print('P')
print(P)

# 3.1.2 Usage Matrix U
rawUsage = [9,0,15,110]
V = np.array(rawUsage).reshape((len(rawUsage), 1))
U = np.column_stack((np.ones((4,1)),V))
print('\nU')
print(U)

# 3.1.3 Bill Matrix B
B = np.matmul(U, P)
print('\nB')
print(B)

####################################################################
# 3.2 - Bill Line Items
####################################################################
Q = P
P = np.zeros((2,2))
np.fill_diagonal(P, Q)

# Recalculate Bill
B32 = np.matmul(U, P)
print('\nB32')
print(B32)

####################################################################
# 3.3 - Tiers
####################################################################
# Q
tiers = [0,10,100]
prices = [2,1,0.5]
Q_tiers = np.column_stack([tiers,prices])
print('\nQ_tiers')
print(Q_tiers)

P_tiers = np.zeros((3, 3), float)
np.fill_diagonal(P_tiers, prices)
print('\nP_tiers')
print(P_tiers)

# Usage Matrix: Row per account and Column per tier
print('\nrawUsage')
print(rawUsage)

tiers.append(np.inf)
arrP_tiers = np.array(tiers) # and inf for last bucket
print('\narrP_tiers')
print(arrP_tiers)
U_tiers = np.array([[x if lo<x<hi else 0 for lo, hi in zip(arrP_tiers[:-1], arrP_tiers[1:])] for x in rawUsage])

print('\nU_tiers')
print(U_tiers)

# Calc Bill, B
B_tiers = np.matmul(U_tiers, P_tiers)
print('\nB_tiers')
print(B_tiers)

####################################################################
# 3.4 - More Complexity:
# Volume, Stairstep, Minimum Spend, Segmented pricing etc.
####################################################################

# Minimum Spend
M = np.concatenate((B_tiers, np.maximum(18-B_tiers.sum(axis=1, keepdims=True),0)), axis=1)
print('\nM')
print(M)

# 3.5.1 Pricing for Multiple Accounts
rawPrice.extend(prices)
P_multi = np.zeros((5, 5), float)
np.fill_diagonal(P_multi, rawPrice)
print('\nP_multi')
print(P_multi)

# 3.5.2 Multi Plan Usage
U_multi = np.zeros((4,5))
U3 = U[0,]
U4 = U_tiers[1:4,:]
U_multi[0, 0:2] = U3
U_multi[1:, 2:] = U4
print('\nU_multi')
print(U_multi)

# Calc Bill for multi plans
B_multi = np.matmul(U_multi, P_multi)
print('\nB_multi')
print(B_multi)

P3 = B_multi.sum(axis=1, keepdims=True)
print('\nP3')
print(P3)

# TODO Check that P3 was solved for numerically


#######################################################################################################################################
# Simulate 5,000 accounts, 36 months, and 10,000 price points
#######################################################################################################################################
rows = np.repeat(np.arange(0, (5000 * 36)), 100)
cols = np.random.randint(1,10000,(5000 * 36 * 100))
data = uniform.rvs(size=len(rows), loc = 0)

Uperf = sparse.coo_matrix((data, (rows, cols)))
diagonals = uniform.rvs(size=Uperf.shape[1], loc = 0)
Pperf = diags(diagonals)

#######################################################################################################################################
# Code for timing
#######################################################################################################################################
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'\nFunction {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

@timeit
def calculate_U_dot_P(U, P):
    #print(U.shape)
    #print(P.shape)
    B = U.dot(P)
    return B

# Calc Bill
Bperf = calculate_U_dot_P(Uperf, Pperf)

#######################################################################################################################################
# Space Efficiency
#######################################################################################################################################
print('\nUperf dim: ' + str(Uperf.shape) +
      '\nUperf number of elements: ' + str(Uperf.shape[0] * Uperf.shape[1]) +
      '\nUperf size in MB: ' + str(Uperf.data.nbytes/(1024**2)))

print('\nPperf dim: ' + str(Pperf.shape) +
      '\nPperf number of elements: ' + str(Pperf.shape[0] * Pperf.shape[1]) +
      '\nPperf size in MB: ' + str(Pperf.data.nbytes/(1024**2)))

print('\nBperf dim: ' + str(Bperf.shape) +
      '\nBperf number of elements: ' + str(Bperf.shape[0] * Bperf.shape[1]) +
      '\nBperf size in MB: ' + str(Bperf.data.nbytes/(1024**2)))

