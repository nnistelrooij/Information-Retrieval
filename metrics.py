import numpy as np
from scipy.stats import kendalltau

def precision_at_k(rel, k):
  """
  Compute the precision@k of a ranked list.

  Args:
    rel = [[int]] relevance judgements for the ranked list
    k   = [int] depth at which to compute the precision

  Returns [int]:
    precision@k
  """
  assert k >= 1 and k <= len(rel), "Invalid depth k."
  return np.mean(rel[:k])

def precision_at_k_wrapper(rel, k):
  """
  Wrapper for `precision_at_k(rel, k)` that adds zeros
  to the relevance judgements collection up to `k`.
  """
  if k > len(rel):
    zeros = np.zeros(k - len(rel))
    rel = np.concatenate((rel, zeros))
  return precision_at_k(rel, k)

def mean_precision_at_k(rels, k):
  """
  Compute the mean precision@k from multiple ranked lists.

  Args:
    rels = [[[int]]] collection of ranked lists
    k    = [int] depth at which to compute the precision

  Returns [int]:
    mean precision@k
  """
  precisions = [precision_at_k(rel, k) for rel in rels]
  return np.mean(precisions)

def average_precision(rel, num_rel_docs):
  """
  Compute the average precision of a ranked list.
  The precision@k is 0 when the k-th element is non-relevant.
  From Zhai & Massung Chapter 9, page 177.

  Args:
    rel           = [[int]] relevance judgements for the ranked list
    num_rel_docs  = [int] number of all relevant documents in the collection

  Returns [int]:
    average precision
  """
  N = len(rel)
  precisions = [precision_at_k(rel, k) for k in np.arange(1, N+1) if rel[k-1]]
  avp = np.sum(precisions) / num_rel_docs
  return avp

def mean_average_precision(rels):
  """
  Compute the mean average precision from multiple ranked lists.
  From Zhai & Massung Chapter 9, page 178.

  Args:
    rels = [[tuple]] collection of tuples in the form 
                     (collection of ranked lists, number of all relevant documents in the collection)

  Returns [int]:
    mean average precision
  """
  avps = [average_precision(rel, num_rel_docs) for rel, num_rel_docs in rels]
  return np.mean(avps)

def geometric_mean_average_precision(rels):
  """
  Compute the geometric mean average precision from
  multiple ranked lists in log space.
  From Zhai & Massung Chapter 9, page 179.

  Args:
    rels = [[tuple]] collection of tuples in the form 
                     (collection of ranked lists, number of all relevant documents in the collection)

  Returns [int]:
    geometric mean average precision
  """
  with np.errstate(divide='ignore'):
    avps = [np.log(average_precision(rel, num_rel_docs)) for rel, num_rel_docs in rels]
  return np.exp(np.mean(avps))

def kendall_tau(rankA, rankB):
  """
  Compute Kendall's tau for two ranked lists.
  Uses the `kendalltau` function from the `scipy` Python libary.

  Args:
    rankA = [[str]] ranked list of document identifiers
    rankB = [[str]] ranked list of document identifiers

  Returns [(int)]:
    Tuple of Kendall's tau and p-value
  """
  assert len(rankA) == len(rankB), "Length of the ranked lists must be identical."
  assert set(rankA) == set(rankB), "Ranked lists must contain identical elements."
  N = len(rankA)
  dictA, dictB = {}, {}
  for k in np.arange(N):
    dictA[rankA[k]] = k
    dictB[rankB[k]] = k
  A = [rank for _, rank in sorted(dictA.items())]
  B = [rank for _, rank in sorted(dictB.items())]
  return kendalltau(A, B)  

if __name__ == "__main__":
  # Example from Zhai & Massung Chapter 9, p.174
  # Testing precision@k
  rel = [1, 1, 0, 0, 1, 0, 0, 1]
  result = [precision_at_k(rel, k) for k in np.arange(1, len(rel)+1)]
  print("P@k:", result)
  # Testing average precision
  avp = average_precision(rel, 10)
  print("AP =", avp)
  # Testing MAP and gMAP (no reference answer though)
  rel1 = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
  rel2 = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0]
  rel3 = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
  rel4 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  rels = [rel1, rel2, rel3, rel4]
  num_rel_doc = [10, 8, 8, 12]
  print("MAP =", mean_average_precision(zip(rels, num_rel_doc)))
  print("gMAP =", geometric_mean_average_precision(zip(rels, num_rel_doc)))
  # Testing Kendall's tau (no reference answer though)
  # Some bogus ranked lists:
  # rankA = ['LET-02', 'SEM-31', 'SEM-25', 'LIT-12', 'FUN-42']
  # rankB = ['SEM-25', 'LET-02', 'FUN-42', 'SEM-31', 'LIT-12']
  # rankC = ['LET-02', 'SEM-31', 'SEM-25', 'FUN-42', 'LIT-12']
  rankA = ['D', 'E', 'B', 'A', 'C']
  rankB = ['B', 'C', 'A', 'D', 'E']
  print(kendall_tau(rankA, rankB))
  # Testing mean precision@k
  print("mean P@k =", mean_precision_at_k(rels, 5))
  # Testing precision@k wrapper
  print("P@k wrapper =", precision_at_k_wrapper(rel, 20))
