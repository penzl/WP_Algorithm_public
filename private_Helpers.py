import numpy as np
import multiprocessing
from functools import partial
from scipy.spatial import distance
from numba import jit
from numba.typed import List


def make_dist_matrix(dim, points1, points2, lmbd, z):
    k = 2 * np.pi / lmbd
    length = dim * 2 - 1
    matrix = np.zeros((length, length), dtype=float)
    matrix[:dim * 2 // 2, :dim * 2 // 2] = np.reshape(distance.cdist(np.array([points2[-1]]), points1), (dim, dim))
    matrix[:dim * 2 // 2:, dim * 2 // 2 - 1:] = np.fliplr(matrix[:dim * 2 // 2:, :dim * 2 // 2:])
    matrix[dim * 2 // 2 - 1:, :] = np.flipud(matrix[:dim * 2 // 2, :])
    pars = (1 / matrix) * np.exp(1j * k * matrix) * (1 + z / matrix)
    return pars

def propagation_v2(lenght, pars, dim, Img, indices, ind):
    # Function that calculates one iteration for propFKintegralMultiCore
    [x, y] = indices[ind]
    return np.sum(
        np.multiply(pars[lenght // 2 - x:lenght // 2 - x + dim, lenght // 2 - y:lenght // 2 - y + dim], Img))


@jit(nopython=True, parallel=True)  # Set "nopython" mode for best performance, equivalent to @njit
def propagation_numba_parallel(lst, lenght, pars, dim, Img, indices, out):
    # Function that calculates all iteration for propFKintegralMultiCore and optimized by numba
    for i in lst:
        [x, y] = indices[i]
        out[x, y] = np.sum(
            np.multiply(pars[lenght // 2 - x:lenght // 2 - x + dim, lenght // 2 - y:lenght // 2 - y + dim], Img))
    return out


def prop_fk_integral_v2(Mask, points1, points2, Img, lmbd, dim, z, MultiCore=False,
                        Symmetrize=False, numba=False):
    # Propagation function based on the Fersnel-Kirchoff formula (see
    # Appl. Phys. Lett. 112, 221104 (2018); https://doi.org/10.1063/1.5027179)
    # "points1" is a list of [x,y,z] points on the source grid (z=0),
    # "points2" is a list of [x,y,z] points on the target grid (z="z"),
    # "Mask" is a 2D array of points that should be included in calculation.
    # "Mask.flatten()" creates a list that corresponds to points2
    # "Img" is a grid ("dim" x "dim") that corresponds to the point sources on
    # the source plane (z=0). Each point is a complex value, representing
    # the amplitude and phase of electromagnetic radiation with the wavelenght "lmdb".
    # MultiCore = kwargs.get('MultiCore', False)
    # Symmetrize = kwargs.get('Symmetrize', False)
    # ProgressBar = kwargs.get('ProgressBar', False)
    pars = make_dist_matrix(dim, points1, points2, lmbd, z)

    out = np.zeros((dim, dim), dtype='complex_')

    indices = np.indices((dim, dim)).T.reshape(-1, 2)
    lenght = dim * 2 - 1
    lst = np.where(Mask.flatten() != 0.0)[0]

    if Symmetrize:
        lst = common_elements(np.where(Mask.flatten() != 0.0)[0], list_of_symmetric_elements(dim))
        print("\tSymmetric calculation!")
        print("   ....")

    if MultiCore:
        out_temp = out.flatten()
        pools = int(multiprocessing.cpu_count() * 0.75)  # lets leave two cpus free ...
        print("\tUsing " + str(pools) + " cores!")
        func = partial(propagation_v2, lenght, pars, dim, Img, indices)
        with multiprocessing.Pool(pools) as pool:
            out0 = pool.map(func, lst)
        out_temp[lst] = out0
        out = np.transpose(np.reshape(out_temp, (dim, dim)))

    if numba:
        print("Using numba optimization...")
        typed_lst = List()
        [typed_lst.append(x) for x in lst]
        # out = Propagation_numba(typed_lst, lenght, pars, dim, Img, indices, out)
        out = propagation_numba_parallel(typed_lst, lenght, pars, dim, Img, indices, out)
    else:
        for i in lst:
            [x, y] = indices[i]
            out[x, y] = propagation_v2(lenght, pars, dim, Img, indices, i)

    out = out * (-1j / (2 * lmbd))

    if Symmetrize:
        out[0:int(len(out) / 2), int(len(out) / 2):] = np.fliplr(out[0:int(len(out) / 2), 0:int(len(out) / 2)])
        out[int(len(out) / 2):, :] = np.flipud(out[0:int(len(out) / 2), :])

    return out
