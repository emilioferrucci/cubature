from utils import tensorSum, xi, gaussianCubature, epsilon, pi, tensorSum
from free_lie_algebra import lieProduct, exp, Elt
from numpy import sqrt
import numpy as jnp
import pickle
from pathlib import Path




def degree_3(dimension : int = 3, drift : bool = True):
    z = gaussianCubature(degree = 3, dimension = 3)
    cubature = []
    for z_n, z_weight in z:
        L = xi([0])
        L += tensorSum([
                    z_n[i] * xi([i])
                    for i in range(1,dimension + 1)
                ])
        cubature.append((L, z_weight))
        
    return cubature


def degree_5(dimension : int = 3, drift : bool = True):
  x = 0 # constant used in Lyons-Victoir construction (assumes any value so taken to be zero for simplicity)

  z, gauss_lam = gaussianCubature(degree = 5, dimension = dimension) # degree 5 cubature on Gaussian

  e = [epsilon(i) for i in range(dimension + 1)] # tensor space basis

  lie_poly = [] # initiate lie_poly
  lam = [] # initiate lam
  cubature = []
  
  for k, zk in enumerate(z):
    sum1 = tensorSum([(1/12)*(zk[i-1])**2*lieProduct(lieProduct(e[0],e[i]), e[i]) for i in range(1,dimension+1)])
    sum2 = tensorSum([zk[i-1]*e[i] for i in range(1, dimension+1)])
    sum3 = tensorSum([tensorSum([(1/2)*zk[i-1]*zk[j-1]*lieProduct(e[i],e[j]) for j in range(i+1,dimension+1)]) for i in range(1,dimension+1)])
    sum4 = tensorSum([tensorSum([(1/6)*zk[i-1]*zk[j-1]**2*lieProduct(lieProduct(e[i],e[j]),e[j]) for j in range(i+1,dimension+1)]) for i in range(1,dimension+1)])
    sum5 = tensorSum([tensorSum([(1/6)*zk[j-1]*zk[i-1]**2*lieProduct(lieProduct(e[j],e[i]),e[i]) for j in range(i+1,dimension+1)]) for i in range(1,dimension+1)])

    cubature.append((e[0]+sum1+sum2+sum3+x*sum4+(1-x)*sum5, (1/2)*gauss_lam[k]))
    cubature.append((e[0]+sum1+sum2-sum3+(1-x)*sum4+x*sum5, (1/2)*gauss_lam[k]))

  return cubature


def degree_7(dimension : int = 3, drift : bool = True):
    """Computes the degree-seven cubature formula for Brownian motion given in Theorem 5.8"""
    
    z = gaussianCubature(degree = 7, dimension = dimension)
    zz = gaussianCubature(degree = 3, dimension = int(dimension * (dimension + 1) / 2))

    def dimensionalise(L : list[float]):
        empty = [[None for _ in range(dimension + 1)] for _ in range(dimension + 1)]
        L = iter(L)
        for i in range(1, dimension + 1):
            for j in range(i, dimension + 1):
                empty[i][j] = next(L)
        return empty
    
    zz = [(dimensionalise(x[0][1:]), x[1]) for x in zz]
    zzz = gaussianCubature(degree = 3, dimension = 1)

    cubature = []
    for z_n, z_weight in z:
        for z_m, zz_weight in zz:
            for z_r, zzz_weight in zzz:
                L = tensorSum([
                    z_n[i] * xi([i])
                    for i in range(1,dimension + 1)
                ])
                L += tensorSum([
                    (1 / sqrt(6)) * z_m[i][i] * z_n[j] * z_r[1] * xi([i,j,i])
                    for i in range(1, dimension + 1) for j in range(1, dimension + 1)
                ])
                L += tensorSum([
                    (1 / sqrt(3)) * (z_m[i][j] + z_m[i][i] * z_n[j] + c * z_m[j][j] * z_n[i]) * xi([i,j])
                    + (1 / 2) * z_n[i] * xi([i,j,j])
                    + (1 / 2) * z_n[j] * xi([j,i,i])
                    for i,j,c in [(1,2,-1), (1,3,1), (2,3,1)]
                ])
                L += tensorSum([
                    (1 / sqrt(3)) * z_m[i][j] * z_n[k] * z_r[1] * xi([i,k,j])
                    + (c / (2 * sqrt(3))) * z_n[i] * z_m[j][j] * xi([i,j,k,k])
                    - (c / (2 * sqrt(3))) * z_n[i] * z_m[j][j] * xi([j,i,k,k])
                    + (c / (2 * sqrt(3))) * z_n[i] * z_m[j][j] * xi([i,k,k,j])
                    for i,j,c in [(1,2,-1), (1,3,1), (2,3,1)] for k in range(1, dimension + 1)
                ])
                L += tensorSum([
                    + (1 / 12) * z_n[k] * xi([k,i,i,j,j])
                    + (1 / 24) * z_n[k] * xi([i,i,k,j,j])
                    for i in range(1, dimension + 1) for j in range(1, dimension + 1) for k in range(1, dimension + 1)
                ])
                if drift:
                    L += xi([0])
                    L += tensorSum([
                        sqrt(1/3) * c * z_m[i][i] * xi([0,i])
                        + (1/2) * xi([0,i,i])
                        for i, c in [(1,-1),(2,-1),(3,1)]
                    ])
                    L += tensorSum([
                        (1/12) * xi([0,i,i,j,j])
                        + (1/24) * xi([i,i,0,j,j])
                        for i in range(1,dimension + 1) for j in range(1,dimension + 1)
                    ])
                
                cubature.append((L, z_weight * zz_weight * zzz_weight))
    return cubature


def cubature_points(degree : int):
    assert degree in [3,5,7]

    if degree == 3:
        cubature = degree_3()
    elif degree == 5:
        cubature = degree_5()
    elif degree == 7:
        cubature = degree_7()
    
    dimension = 3
    n_points = len(cubature)
    array = []
    for level_val in range(8):
        array.append(jnp.full((n_points,) + (dimension + 1,) * level_val, 0.) )

    for index, (lie_poly, weight) in enumerate(cubature):
        exponentiated_lie_poly = exp(lie_poly, maxLevel = 7)
        for level_val, level_data in enumerate(exponentiated_lie_poly.data):
            if level_val >= 1:
                
                for elt, val in level_data.items():
                    #array = array.at[elt.letters].set(val) ##USE THIS FOR JAXNUMPY
                    array[level_val][index][elt.letters] += val

    return array, jnp.array([x[1] for x in cubature], dtype = jnp.float64)



cubature_3 = cubature_points(3)
cubature_5 = cubature_points(5)
cubature_7 = cubature_points(7)

save_path_3 = Path(__file__).parent / "degree_30.pkl"
save_path_5 = Path(__file__).parent / "degree_50.pkl"
save_path_7 = Path(__file__).parent / "degree_70.pkl"


with open(save_path_3, "wb") as f:
    pickle.dump(cubature_3, f)

with open(save_path_5, "wb") as f:
    pickle.dump(cubature_5, f)

with open(save_path_7, "wb") as f:
    pickle.dump(cubature_7, f)