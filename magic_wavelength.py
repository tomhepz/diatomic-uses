# %%

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

import diatomic
from diatomic.systems import SingletSigmaMolecule
import diatomic.plotting as plotting
import diatomic.operators as operators
import diatomic.calculate as calculate

# %%

# Define helpful constants
pi = scipy.constants.pi
bohr = scipy.constants.physical_constants["Bohr radius"][0]
eps0 = scipy.constants.epsilon_0

GAUSS = 1e-4  # T
MHz = scipy.constants.h * 1e6
kHz = scipy.constants.h * 1e3
muN = scipy.constants.physical_constants["nuclear magneton"][0]
H_BAR = scipy.constants.hbar
kWpercm2 = 1e7


# Set logging
diatomic.configure_logging()

# Generate Molecule
mol = SingletSigmaMolecule.from_preset("Rb87Cs133")
mol.Nmax = 3

# 0.2V
INTEN1065 = 8 * kWpercm2
B = 181.699 * GAUSS

MAGIC817_RATIO = - mol.a02[1065][1]/mol.a02[817][1]
INTEN817_MAGIC = MAGIC817_RATIO * INTEN1065


# %%

LEADING_STEPS = 10
INTEN817_DESIRED_STEPS = 31 # Want to be odd to include magic middle point
INTEN817_STEPS = LEADING_STEPS + INTEN817_DESIRED_STEPS
INTEN817_MIN = 0.9 * INTEN817_MAGIC
INTEN817_MAX = 1.1 * INTEN817_MAGIC
MAGIC_INDEX = LEADING_STEPS + int((INTEN817_DESIRED_STEPS-1)/2)


INTEN817 = np.concatenate((np.linspace(0, INTEN817_MIN, LEADING_STEPS),
                           np.linspace(INTEN817_MIN, INTEN817_MAX, INTEN817_DESIRED_STEPS))
                           )
print(MAGIC_INDEX)
print(INTEN817[MAGIC_INDEX], INTEN817_MAGIC)

# %%

# Generate Hamiltonians
H0 = operators.hyperfine_ham(mol)
Hz = operators.zeeman_ham(mol)

# NOTE: Tweezer polarisations likely are fractionally eliptical on either side of the trap
Hac1065 = operators.ac_ham(mol, a02=mol.a02[1065], beta=0)
Hac817  = operators.ac_ham(mol, a02=mol.a02[817], beta=0)

# Overall Hamiltonian
Htot = H0 + Hz * B + Hac1065 * INTEN1065 + Hac817 * INTEN817[:, None, None]

# Solve (diagonalise) Hamiltonians
eigenergies, eigstates = calculate.solve_system(Htot)

eiglabels = calculate.label_states(mol, eigstates[0], ["N", "MF"], index_repeats=True)


def label_to_indices(labels, N, MF):
    labels = np.asarray(labels)
    indices = np.where((labels[:, 0] == N) & (labels[:, 1] == MF))[0]
    return indices


rovibgroundstate = label_to_indices(eiglabels, 0, 5)[0]
one_six_zero = label_to_indices(eiglabels, 1, 6)[0]
one_five_zero = label_to_indices(eiglabels, 1, 5)[0]
one_four_zero = label_to_indices(eiglabels, 1, 4)[0]

pi_coupling = calculate.transition_electric_moments(
    mol, eigstates, 0, #from_states=[0,rovibgroundstate,one_six_zero]
)
sigma_plus_coupling = calculate.transition_electric_moments(
    mol, eigstates, +1, #from_states=[0,rovibgroundstate,one_six_zero]
)
sigma_minus_coupling = calculate.transition_electric_moments(
    mol, eigstates, -1, #from_states=[0,rovibgroundstate,one_six_zero]
)

# %%
fig, (ax) = plt.subplots(constrained_layout = True)

# (0,32)
# (32,128)
# (128,288)

# coupling_matrices = (pi_coupling, sigma_plus_coupling, sigma_minus_coupling)
# reference_state_energy = eigenergies[:, rovibgroundstate]
# # for ax, sl, su, ylim in ((axl,0,32,(-980.5,-979.75)),(axu,128,288,(1960.4,1961.1))):
# for i in range(32, 128):
#     transition_energy = (eigenergies[:, i] - reference_state_energy) - (eigenergies[MAGIC_INDEX, i] - reference_state_energy[MAGIC_INDEX])
#     eiglabel = eiglabels[i]

#     ax.plot(INTEN817 / kWpercm2, transition_energy / kHz, c="k", alpha=0.01)
#     pol_index = (eiglabel[1] - eiglabels[rovibgroundstate][1])
    
#     if pol_index in (-1,0,1):
#         coupling_matrix = coupling_matrices[pol_index]

#         alpha_values = (coupling_matrix[:, 1, i] / (mol.d0)) ** 0.5
#         colors = np.zeros((INTEN817_STEPS, 4))
#         if pol_index == 0:
#             c_on = 2
#         elif pol_index == 1:
#             c_on = 1
#         else:
#             c_on = 0
#         colors[:, c_on] = 1  # red
#         colors[:, 3] = alpha_values
#         plotting.colorline(
#             ax,
#             INTEN817 / kWpercm2,
#             transition_energy / kHz,
#             colors,
#             linewidth=1.5,
#         )

coupling_matrices = (pi_coupling, sigma_plus_coupling, sigma_minus_coupling)
reference_state_index = label_to_indices(eiglabels, 1, 6)[0]
reference_state_energy = eigenergies[:, reference_state_index]
for i in range(128, 288):
    transition_energy = (eigenergies[:, i] - reference_state_energy) - (eigenergies[MAGIC_INDEX, i] - reference_state_energy[MAGIC_INDEX])
    eiglabel = eiglabels[i]

    ax.plot(INTEN817 / kWpercm2, transition_energy / kHz, c="k", alpha=0.01)
    pol_index = (eiglabel[1] - eiglabels[reference_state_index][1])
    
    if pol_index in (-1,0,1):
        coupling_matrix = coupling_matrices[pol_index]

        alpha_values = (coupling_matrix[:, reference_state_index, i] / (mol.d0)) ** 0.5
        colors = np.zeros((INTEN817_STEPS, 4))
        if pol_index == 0:
            c_on = [0,1]
        elif pol_index == 1:
            c_on = [0,2]
        else:
            c_on = [1,2]
        colors[:, c_on] = 1
        colors[:, 3] = alpha_values
        plotting.colorline(
            ax,
            INTEN817 / kWpercm2,
            transition_energy / kHz,
            colors,
            linewidth=1.5,
        )


ax.axvline(INTEN817_MAGIC/ kWpercm2, c='k', linestyle='--')

ax.set_xlim(0.9 * INTEN817_MAGIC/ kWpercm2, 1.1 * INTEN817_MAGIC/ kWpercm2)
ax.set_ylabel(r"$(0,5)_0 \rightarrow (N=1, MF)_k$  deviation from magic detuning (kHz)")
# ax.set_ylim(-10,10)

ax.set_title(fr"$I_{{1065}}$ =  {INTEN1065/kWpercm2} kWcm$^{{-2}}$")

ax.set_xlabel("817nm Intensity ($kW/cm^2$)")

# print(sigma_plus_coupling[0,0,:])

plt.show()
# input("Enter to close")

# %%
