from free_lie_algebra import word2Elt, Word, Elt, lieProduct, concatenationProduct, exp, concatenationProductMany
from numpy import sqrt
import numpy as np
from itertools import product, permutations

def epsilon(index : int):
    """Returns canonical basis element in the tensor algebra for given index"""
    return word2Elt(Word([index]))

def xi(index : list[int]):
    """Helper function that returns Eulerian idopotent from given word [given as an integer list], using PWB basis representation for efficient computation."""
    l = len(index)
    index   = [epsilon(x) for x in index]

    if l == 1:
        return index[0]
    elif l == 2:
        return 1 / 2 * lieProduct(index[0], index[1])
    elif l == 3:
        return 1 / 6 * (lieProduct(index[0],lieProduct(index[1],index[2])) + lieProduct(lieProduct(index[0],index[1]), index[2]))
    elif l == 4:
        return 1 / 12 * (
            lieProduct(index[0],lieProduct(lieProduct(index[1],index[2]),index[3]))
            + lieProduct(lieProduct(index[0],index[1]),lieProduct(index[2],index[3]))
            + lieProduct(lieProduct(index[0],index[2]),lieProduct(index[1],index[3]))
            + lieProduct(lieProduct(index[0],lieProduct(index[1],index[2])),index[3])
        )
    elif l == 5:
        return 1/60 * (
            -2 * lieProduct(index[0],lieProduct(index[1],lieProduct(index[2],lieProduct(index[3],index[4]))))
            + lieProduct(lieProduct(index[0],lieProduct(index[1],lieProduct(index[2],index[3]))),index[4])
            + 3 * lieProduct(index[0],lieProduct(index[1],lieProduct(lieProduct(index[2],index[3]),index[4])))
            + 3 * lieProduct(index[0],lieProduct(lieProduct(index[1],index[2]),lieProduct(index[3],index[4])))
            + 3 * lieProduct(lieProduct(lieProduct(index[0],index[1]),lieProduct(index[2],index[3])),index[4])
            + 3 * lieProduct(index[0],lieProduct(lieProduct(index[1],index[3]),lieProduct(index[2],index[4])))
            + 3 * lieProduct(lieProduct(lieProduct(index[0],index[2]),lieProduct(index[1],index[3])),index[4])
            + 3 * lieProduct(lieProduct(lieProduct(index[0],lieProduct(index[1],index[2])),index[3]),index[4])
            + lieProduct(index[0],lieProduct(lieProduct(lieProduct(index[1],index[2]),index[3]),index[4]))
            - 2 * lieProduct(lieProduct(lieProduct(lieProduct(index[0],index[1]),index[2]),index[3]),index[4])
            + lieProduct(lieProduct(index[0],index[1]),lieProduct(index[2],lieProduct(index[3],index[4])))
            + lieProduct(lieProduct(index[0],index[2]),lieProduct(index[1],lieProduct(index[3],index[4])))
            + lieProduct(lieProduct(index[0],index[3]),lieProduct(index[1],lieProduct(index[2],index[4])))
            + 2 * lieProduct(lieProduct(index[0],lieProduct(index[1],index[4])),lieProduct(index[2],index[3]))
            + 2 * lieProduct(lieProduct(index[0],lieProduct(index[2],index[4])),lieProduct(index[1],index[3]))
            + 2 * lieProduct(lieProduct(index[0],lieProduct(index[3],index[4])),lieProduct(index[1],index[2]))
            + lieProduct(lieProduct(lieProduct(index[0],index[1]),index[2]),lieProduct(index[3],index[4]))
            + lieProduct(lieProduct(lieProduct(index[0],index[1]),index[3]),lieProduct(index[2],index[4]))
            - 2 * lieProduct(lieProduct(lieProduct(index[0],index[1]),index[4]),lieProduct(index[2],index[3]))
            + lieProduct(lieProduct(lieProduct(index[0],index[2]),index[3]),lieProduct(index[1],index[4]))
            - 2 * lieProduct(lieProduct(lieProduct(index[0],index[2]),index[4]),lieProduct(index[1],index[3]))
            - 2 * lieProduct(lieProduct(lieProduct(index[0],index[3]),index[4]),lieProduct(index[1],index[2]))
        )
    
def symmetricProduct(l):
  perms = list(set(permutations(l)))
  total = Elt([{}])
  for i, perm in enumerate(perms):
    total += concatenationProductMany(list(perm))
  return (1/len(perms))*total

def tensorSum(tensors : list[Elt]):
    sum = Elt([])
    for tensor in tensors:
        sum += tensor
    return sum

def pi(tensor : Elt, degree : int = 7):
  """Projects tensor into the truncated tensor algebra with given degree"""
  elt_data = tensor.data[:2*degree+1]
  for i in range(len(elt_data)):
    i_data = elt_data[i]
    for key in i_data.keys():
      total_d = i + key.letters.count(0)
      if total_d > degree:
        tensor -= Elt([{}]*i+[{key : i_data[key]}])
  return tensor

def expected_signature(degree : int = 7, dimension : int = 3, drift : bool = True):
  """Computes the expected signature of Brownian motion in the tensor algebra"""
  expected_signature = tensorSum([(1 / 2) * concatenationProduct(epsilon(i), epsilon(i)) for i in range(1, dimension + 1)])
  if drift:
    expected_signature += epsilon(0)
  expected_signature = pi(exp(expected_signature, maxLevel = degree), degree)

  return expected_signature

def gaussianCubature(degree : int, dimension : int):
   """Computes a cubature formula for the Gaussian measure, as seen in Stroud"""
   assert degree in [7,5,3]
   if degree == 3:
      return [([None] + [0] * i + [sqrt(dimension)] + [0] * (dimension-i-1) , 1 / (2 * dimension)) for i in range(dimension)] + [([None] + [0] * i + [- sqrt(dimension)] + [0] * (dimension - i - 1), 1 / (2 * dimension)) for i in range(dimension)]
   if degree == 5:
      d = 3
      eta = 0.476731294622796
      lam = 0.935429018879534
      xi = -0.731237647787132
      mu =  0.433155309477649
      gam = 2.66922328697744
      A =   0.242000000000000
      B =   0.081000000000000
      C =   0.005000000000000
      
      z = [] # initiate variables and append values
      weights = []

      z.append([eta]*d)
      weights.append(A)

      z.append([-eta]*d)
      weights.append(A)
      
      for i in range(d):
        z.append([xi]*i+[lam]+[xi]*(d-i-1))
        weights.append(B)

        z.append([-xi]*i+[-lam]+[-xi]*(d-i-1))
        weights.append(B)

        for j in range(i+1, d):
          z.append([gam]*i+[mu]+[gam]*(j-i-1)+[mu]+[gam]*(d-j-1))
          weights.append(C)

          z.append([-gam]*i+[-mu]+[-gam]*(j-i-1)+[-mu]+[-gam]*(d-j-1))
          weights.append(C)

      z = [[np.sqrt(2)*y for y in x] for x in z]
      return z, weights
   elif degree == 7:
      assert dimension == 3
      r = sqrt(15 + sqrt(15)) / 2
      s = sqrt(9 + 2 * sqrt(15)) / sqrt(2)
      t = sqrt(6 - sqrt(15)) / sqrt(2)
      V = np.pi ** (3 / 2)
      B = 1 * (5 / (8 * r ** 6)) 
      C = 1 / (64 * s ** 6)
      D = 1 / (16 * t ** 6)
      A = (1 - 6 * B - 8 * C - 12 * D)
      r = r * sqrt(2)
      s = s * sqrt(2)
      t = t * sqrt(2)

      return [([None, 0, 0, 0], A)] + [([None] + [0] * i + [r] + [0] * (2 - i), B) for i in range(3)] + [([None] + [0] * i + [-r] + [0] * (2 - i), B) for i in range(3)] + [([None] + list(x), C) for x in product((s, -s), repeat = 3)] + [([None] + list(x), D) for x in permutations([t,t,0])][2 : -1] + [([None] + list(x), D) for x in permutations((t,-t,0))] + [([None] + list(x), D) for x in permutations([-t,-t,0])][2 : -1]