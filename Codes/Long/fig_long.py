"""
long_algorithm.py

N must be greater than 4.
"""

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.linalg import fractional_matrix_power

# LaTeX
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

# constants
N = 8
theta = 2 * np.arcsin(1 / np.sqrt(N))
j = int(np.ceil((np.pi / 2 - theta / 2) / theta))
phi = 2 * np.arcsin(np.sin(np.pi / (4 * j + 2)) / (np.sin(theta / 2)))

# states
beta = qt.basis(2, 1)
alpha = qt.basis(2, 0)
psi = np.cos(theta / 2) * alpha + np.sin(theta / 2) * beta

# operation
I_O = qt.Qobj([[1, 0], [0, np.exp(1j * phi)]])
basictrans = qt.Qobj(
    [[np.cos(theta / 2), np.sin(theta / 2)], [np.sin(theta / 2), -np.cos(theta / 2)]]
)
basictrans_inv = basictrans.inv()
I_D = basictrans * qt.Qobj([[1, 0], [0, np.exp(-1j * phi)]]) * basictrans_inv
Q = I_D * I_O

# shaft of rotation
omega, axiss = Q.eigenstates()
if 2 * abs((beta.dag() * axiss[0])) ** 2 > 1:
    axis = axiss[0]
else:
    axis = axiss[1]

# iterational states
psi_O = I_O * psi
psis = [psi]
for idx in range(j - 1):
    psis.append(Q * psis[-1])

# auxiliar states
aux_alpha = np.sqrt(np.cos(theta)) * alpha
aux_psi = np.sqrt(np.cos(2 * np.arcsin(np.sin(theta) * np.sin(phi / 2)))) * psi
aux_axis = np.sqrt(2 * abs((beta.dag() * axis)) ** 2 - 1) * axis

# Initialize the Bloch sphere
bloch = qt.Bloch()

# add states
bloch.add_states(aux_alpha)
vec_aux_alpha = bloch.vectors[-1]
bloch.add_states(aux_psi)
vec_aux_phi = bloch.vectors[-1]
bloch.add_states(aux_axis)
vec_aux_axis = bloch.vectors[-1]

bloch.clear()

bloch.add_states(alpha, alpha=0.7)
vec_alpha = bloch.vectors[-1]
bloch.add_states(psi, alpha=0.7)
vec_psi = bloch.vectors[-1]
vec_psis = [vec_psi]
bloch.add_states(psi_O, alpha=0.7)
vec_psi_O = bloch.vectors[-1]
bloch.add_states(beta, alpha=0.7)
vec_beta = bloch.vectors[-1]
bloch.add_states(axis, alpha=0.7)
vec_axis = bloch.vectors[-1]
for _ in range(1, j):
    bloch.add_states(psis[_], alpha=0.7)
    vec_psis.append(bloch.vectors[-1])


# State of transition
dt = 15
t = np.linspace(0, 1, dt)
for _ in t:
    IO_trans = qt.Qobj([[1, 0], [0, np.exp(1j * phi * _)]])
    psi_trans = IO_trans * psi
    bloch.add_states(psi_trans, kind="point", colors="g", alpha=0.7)
    IW_trans = (
        basictrans * qt.Qobj([[1, 0], [0, np.exp(-1j * phi * _)]]) * basictrans_inv
    )
    psi_O_trans = IW_trans * psi_O
    bloch.add_states(psi_O_trans, kind="point", colors="b", alpha=0.7)
    for idx in range(0, j):
        psis_trans = qt.Qobj(fractional_matrix_power((I_D * I_O).full(), _)) * psis[idx]
        bloch.add_states(psis_trans, kind="point", colors="r", alpha=0.7)


# states labels
bloch.xlabel = ["", ""]
bloch.ylabel = ["", ""]
bloch.zlabel = ["", ""]
bloch.add_annotation(vec_alpha, r"$\left|\alpha\right\rangle$")
bloch.add_annotation(
    vec_beta, r"$Q^{" + str(j) + r"}\left|\psi\right\rangle=\left|\beta\right\rangle$"
)
bloch.add_annotation(vec_psi, r"$\left|\psi\right\rangle$")
bloch.add_annotation(vec_psi_O, r"$I_{O}\left|\psi\right\rangle$")
bloch.add_annotation(
    vec_psis[1], r"$I_{D}I_{O}\left|\psi\right\rangle=Q\left|\psi\right\rangle$"
)
for idx in range(2, j):
    bloch.add_annotation(
        vec_psis[idx], r"$Q^{" + str(idx) + r"}\left|\psi\right\rangle$"
    )
bloch.add_annotation(vec_axis, r"$\left|\tau\right\rangle$")

# colors
colors = [
    "g",  # alpha
    "b",  # psi
    "#0000BF",  # psi_O
    "#000000",  # beta
    "r",  # axis
]
for idx in range(1, j):
    colors.append("#000080")
bloch.vector_color = colors

bloch.figsize = [9, 9]

# lines
bloch.add_points(np.array([vec_aux_alpha, vec_psi]).T, meth="l", colors="g", alpha=0.3)
bloch.add_points(
    np.array([vec_aux_alpha, vec_psi_O]).T, meth="l", colors="g", alpha=0.3
)
bloch.add_points(np.array([vec_aux_phi, vec_psi_O]).T, meth="l", colors="b", alpha=0.3)
bloch.add_points(
    np.array([vec_aux_phi, vec_psis[1]]).T, meth="l", colors="b", alpha=0.3
)
for idx in range(0, j):
    bloch.add_points(
        np.array([vec_psis[idx], vec_aux_axis]).T, meth="l", colors="r", alpha=0.3
    )
bloch.add_points(np.array([vec_beta, vec_aux_axis]).T, meth="l", colors="r", alpha=0.3)

# show
bloch.view = [-78, -4]
bloch.show()

# save
bloch.fig.savefig("Notes/figures/Long.pgf")
