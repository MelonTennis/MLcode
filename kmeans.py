import numpy as np
import math

def update_assignments(X, C):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  # Output:
  #   a is the cluster assignments (n,), 1-d array

  a = np.zeros(X.shape[0])
  for i in range(0, X.shape[0]):
    min_dist = 100000000.0
    for j in range(0, C.shape[0]):
      cur_dist = np.dot(C[j] - X[i], C[j] - X[i])
      if cur_dist < min_dist:
        min_dist = cur_dist
        a[i] = j
  return a

def update_centers(X, C, a):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  # Output:
  #   C is the new cluster centers (k, d), 2-d array

  # cache = [[] for i in range(0, len(a))]
  cache = np.zeros(C.shape)
  count = [0 for i in range(0, C.shape[0])]
  for i in range(0, X.shape[0]):
    cur_min = 100000000.0
    for j in range(0, C.shape[0]):
      cur_dist = np.dot(X[i] - C[j], X[i] - C[j])
      if cur_dist < cur_min:
        cur_min = cur_dist
        a[i] = j
  for k in range(0, len(a)):
    cache[a[k]] += X[k]
    count[int(a[k])] += 1
  for k in range(0, C.shape[0]):
    C[k] = cache[k]/count[k]
  return C



def lloyd_iteration(X, C):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the initial cluster centers (k, d), 2-d array
  # Output:
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array

  a = np.zeros(X.shape[0])
  preC = np.copy(C)
  firstPass = True
  cnt = 0
  while firstPass or (not np.array_equal(preC, C)):
    firstPass = False
    a = update_assignments(X, C)
    preC = np.copy(C)
    C = update_centers(X, C, a)
    cnt += 1
  #a = update_assignments(X, C)
  print cnt
  return (C, a)

def kmeans_obj(X, C, a):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  # Output:
  #   obj is the k-means objective of the provided clustering, scalar, float

  obj = 0.0
  for i in range(0, X.shape[0]):
    obj += np.dot(X[i] - C[a[i]], X[i] - C[a[i]])
  return obj


# kmeans_cluster will be used in the experiments, it is available after you 
# have implemented lloyd_iteration and kmeans_obj.

def kmeans_cluster(X, k, init, num_restarts):
  n = X.shape[0]
  # Variables for keeping track of the best clustering so far
  best_C = None
  best_a = None
  best_obj = np.inf
  for i in range(num_restarts):
    if init == "random":
      perm = np.random.permutation(range(n))
      C = np.copy(X[perm[0:k]])
    elif init == "kmeans++":
      C = kmpp_init(X, k)
    elif init == "fixed":
      C = np.copy(X[0:k])
    else:
      print "No such module"
    # Run the Lloyd iteration until convergence
    (C, a) = lloyd_iteration(X, C)
    # Compute the objective value
    obj = kmeans_obj(X, C, a)
    if obj < best_obj:
      best_C = C
      best_a = a
      best_obj = obj
  return (best_C, best_a, best_obj)


def kmpp_init(X, k):
  n = X.shape[0]
  sq_distances = np.ones(n)
  center_ixs = list()
  for j in range(k):
    # Choose a new center index using D^2 weighting
    ix = discrete_sample(sq_distances)
    # Update the squared distances for all points
    deltas = X - X[ix]
    for i in range(n):
      sq_dist_to_ix = np.power(np.linalg.norm(deltas[i], 2), 2)
      sq_distances[i] = min(sq_distances[i], sq_dist_to_ix)
    # Append this center to the list of centers
    center_ixs.append(ix)
  # Output the chosen centers
  C = X[center_ixs]
  return np.copy(C)


def discrete_sample(weights):
  total = np.sum(weights)
  t = np.random.rand() * total
  p = 0.0
  for i in range(len(weights)):
    p = p + weights[i];
    if p > t:
      ix = i
      break
  return ix