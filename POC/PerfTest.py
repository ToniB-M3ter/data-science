from scipy import sparse
from scipy.stats import uniform
from scipy.sparse import diags
from scipy.sparse import coo_array
import numpy as np
from functools import wraps
import time


# COO Marix
# row indices
row_ind = np.array([0, 1, 1, 3, 4])
# column indices
col_ind = np.array([0, 2, 4, 3, 4])
# data to be stored in COO sparse matrix
data = np.array([1, 2, 3, 4, 5], dtype=float)

# create COO sparse matrix from three arrays
mat_coo = sparse.coo_matrix((data, (row_ind, col_ind)))
# print coo_matrix
print(mat_coo)
print(mat_coo.toarray())

# convert COO to CSC sparse
print(mat_coo.tocsc())

# Create a sparse matrix from a dense matrix
np.random.seed(seed=42)
data = uniform.rvs(size=16, loc = 0, scale=2)
data = np.reshape(data, (4, 4))
print(data)

# make elements with value less < 1 to zero
data[data < 1] = 0
print('data')
print(data)

data_csr = sparse.csr_matrix(data)

# Memory efficiency of sparse matrix
np.random.seed(seed=42)
data = uniform.rvs(size=1000000, loc = 0, scale=2)
# plt.hist(data, 50, facecolor='green')
# plt.show()
data = np.reshape(data, (10000, 100))


data[data < 1] = 0
data_size = data.nbytes/(1024**2)
print('Size of full matrix with zeros: '+ '%3.2f' %data_size + ' MB')


#######################################################################################################################################
# Simulate 5,000 accounts, 36 months, and 10,000 price points
#######################################################################################################################################
rows = np.repeat(np.arange(0, (5000 * 36)), 100) # len(rows) = 18,000,000
cols = np.random.randint(1,10000,(5000 * 36 * 100)) # len(cols) = 18,000,000
data = uniform.rvs(size=len(rows), loc = 0)

Uperf = sparse.coo_matrix((data, (rows, cols)))
Uper_coo = coo_array((data, (rows, cols)), shape=(len(rows), len(cols)))
#print(Uperf.shape, Uper_coo.shape) # (180,000, 10,000) (18,000,000 , 18,000,000)
diagonals = uniform.rvs(size=(5000*36), loc = 0)
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
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

@timeit
def calculate_U_dot_P(U, P):
    print(U.shape)
    print(P.shape)
    # Multiply csr matrix with csc matrix
    B = U.multiply(P)
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






