%matplotlib qt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

def sort_smooth_generalised(eigvecs):
    """
    Smooths the eigenvalue trajectories as a function of an external parameter, rather
    than by energy order, as is the case with `np.eigh`.

    This function sorts the eigenvalues and eigenvectors to maintain continuity as the
    external parameter changes, preventing sudden jumps in the eigenvalue ordering.

    Args:
        eigvals (np.ndarray): Eigenvalues array with shape (steps, eigstate).
        eigvecs (np.ndarray): Eigenvectors array with shape (steps, basis, eigstate).
        num_diagonals (int | None, optional): The number of local swaps above and below
            to consider when smoothing eigenfunctions. Changing this to a number can
            speed up smoothing calculations for large numbers of eigenstates. If None,
            compare all eigenstates to all other eigenstates when smoothing (safest).

    Returns:
        tuple: The smoothed eigenvalues and eigenvectors.

    Raises:
        ValueError: If eigvecs does not have three dimensions.
    """
    basis_size = eigvecs.shape[-2]
    eigstate_count = eigvecs.shape[-1]

    param_step_counts = eigvecs.shape[:-2]
    param_count = len(param_step_counts)

    for param_index in range(param_count):

        param_step_count = param_step_counts[param_index]

        # Compute overlap matrix between (k)th and (k-1)th

        slices_a = (slice(None,-1,1) if pi == param_index else slice(None) for pi in range(param_count))
        slices_b = (slice(1,None,1) if pi == param_index else slice(None) for pi in range(param_count))
        print(slices_a)
        print(slices_b)

        overlap_matrices = np.abs(eigvecs[slices_a].swapaxes(-1, -2) @ eigvecs[slices_b].conj())
        best_overlaps = np.argmax(overlap_matrices, axis=-2)

        # Cumulative permutations
        integrated_permutations = np.empty((*param_step_counts, eigstate_count), dtype=int)
        slices_c = (-1 if pi == param_index else slice(None) for pi in range(param_count))
        integrated_permutations[slices_c] = np.arange(eigstate_count)

        for i in range(param_step_count - 2, -1, -1):
            slices_c = (i if pi == param_index else slice(None) for pi in range(param_count))
            slices_d = (i+1 if pi == param_index else slice(None) for pi in range(param_count))
            integrated_permutations[slices_c] = best_overlaps[slices_c][integrated_permutations[slices_d]]

        # Rearrange to maintain continuity
        # eigvals = eigvals[
        #     np.arange(param_step_count)[:, None], integrated_permutations
        # ]
        
        # eigvecs = eigvecs[
        #     np.arange(param_step_count)[:,   None, None],
        #           np.arange(basis_size)[None, :,   None],
        #         integrated_permutations[:,    None, :  ],
        # ]

        
        # integrated_permutations = np.empty((param_step_count, eigstate_count), dtype=int)
        # integrated_permutations[-1] = np.arange(eigstate_count)

        # for i in range(param_step_count - 2, -1, -1):
        #     integrated_permutations[i] = best_overlaps[i][integrated_permutations[i + 1]]

        # # Rearrange to maintain continuity
        # eigvals = eigvals[
        #     np.arange(param_step_count)[:, None], integrated_permutations
        # ]
        # eigvecs = eigvecs[
        #     np.arange(param_step_count)[:, None, None],
        #     np.arange(basis_size)[None, :, None],
        #     integrated_permutations[:, None, :],
        # ]

    return eigvecs

H0=np.array([[0,0],
            [0,1]])

H1=np.array([[0,0],
            [0,-0.8]])

H2=np.array([[0,0],
            [0,-0.6]])

param1 = np.linspace(0,2,21)

param2 = np.linspace(0,2,20)

X,Y = np.meshgrid(param1,param2,indexing='ij')

Htot = (
    H0 
    + H1 * param1[:,None,None,None]
    + H2 * param2[None,:,None,None]
)

print(Htot.shape)

eigvals, eigvecs = np.linalg.eigh(Htot)

print(eigvals.shape)

eigvecs = sort_smooth_generalised(eigvecs)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

for eig_index in range(eigvals.shape[-1]):
    surf = ax.plot_surface(X, Y, eigvals[:,:,eig_index], linewidth=0, antialiased=False)