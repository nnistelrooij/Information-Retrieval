import numpy as np

def precision_at_k(rel, k):
  """
  Compute the precision@k of a ranked list.

  Args:
    rel = [[int]] relevance judgements for the ranked list
    k   = [int] depth at which to compute the precision

  Returns [int]:
    precision@k
  """
  assert k >= 1 and k <= len(rel), "Invalid depth k"
  return np.mean(rel[:k])

def average_precision(rel):
  """
  Compute the average precision of a ranked list.
  The precision@k is 0 when the k-th element is non-relevant.
  From Zhai & Massung Chapter 9, page 177.

  Args:
    rel = [[int]] relevance judgements for the ranked list

  Returns [int]:
    average precision
  """
  N = len(rel)
  precisions = [precision_at_k(rel, k) for k in np.arange(1, N+1) if rel[k-1]]
  avp = np.sum(precisions) / N
  return avp

def mean_average_precision(rels):
  """
  Compute the mean average precision from multiple ranked lists.
  From Zhai & Massung Chapter 9, page 178.

  Args:
    rels = [[[int]]] collection of ranked lists

  Returns [int]:
    mean average precision
  """
  avps = [average_precision(rel) for rel in rels]
  return np.mean(avps)

def geometric_mean_average_precision(rels):
  """
  Compute the geometric mean average precision from
  multiple ranked lists in log space.
  From Zhai & Massung Chapter 9, page 179.

  Args:
    rels = [[[int]]] collection of ranked lists

  Returns [int]:
    geometric mean average precision
  """
  avps = [np.log(average_precision(rel)) for rel in rels]
  return np.exp(np.mean(avps))

if __name__ == "__main__":
  # Example from Zhai & Massung Chapter 9, p.174
  # Testing precision@k
  rel = [1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  result = [precision_at_k(rel, k) for k in np.arange(1, len(rel)+1)]
  print(result)
  # Testing average precision
  avp = average_precision(rel)
  print(avp)
  # Testing MAP and gMAP (no reference answer though)
  rel2 = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0]
  rel3 = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
  rel4 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  rels = [rel, rel2, rel3, rel4]
  print(mean_average_precision(rels))
  print(geometric_mean_average_precision(rels))
