"""
Ismael_algorithm.py

"""

import matplotlib.pyplot as plt
import numpy as np

# LaTeX
plt.rc("text", usetex=True)

# constants
N = 6
theta = 2 * np.arcsin(1 / np.sqrt(N))
j = int(np.ceil((np.pi / 2 - theta / 2) / theta))
omega = 2 * np.arcsin(1 / np.sqrt(N - 1))
phi = 0

# states
beta = np.array([0, 0, 1])
alpha = np.array([0, 1, 0])
gamma = np.array([1, 0, 0])
psi = np.cos(theta / 2) * alpha + np.sin(theta / 2) * beta
psi_perp = -np.sin(theta / 2) * alpha + np.cos(theta / 2) * beta
psi_ = np.cos(omega / 2) * gamma + np.sin(omega / 2) * psi_perp

# operation
Oracle = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
basictrans_W = np.array(
    [
        [1, 0, 0],
        [0, np.cos(theta / 2), -np.sin(theta / 2)],
        [0, np.sin(theta / 2), np.cos(theta / 2)],
    ]
)
basictrans_W_inv = np.linalg.inv(basictrans_W)
cW_diag = np.array([-1, 1, -1])
cW = np.dot(np.dot(basictrans_W, np.diag(cW_diag)), basictrans_W_inv)
basictrans_Y = np.array(
    [
        [np.cos(theta / 2), 0, np.sin(theta / 2)],
        [0, 1, 0],
        [-np.sin(theta / 2), 0, np.cos(theta / 2)],
    ]
)
basictrans_V = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# iterational states
psi_O = np.dot(Oracle, psi)
psi_WO = np.dot(cW, psi_O)


# vectors
X, Y, Z = np.meshgrid(
    np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)
)
Z_beta = np.outer(beta, np.ones(10))
Z_alpha = np.outer(alpha, np.ones(10))
Z_psi = np.outer(psi, np.ones(10))

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# add states
ax.quiver(0, 0, 0, *beta, color="r")
ax.quiver(0, 0, 0, *alpha, color="g")
ax.quiver(0, 0, 0, *gamma, color="k")
ax.quiver(0, 0, 0, *psi, color="b")
ax.quiver(0, 0, 0, *psi_, color="b")
ax.quiver(0, 0, 0, *psi_O, color="b")
ax.quiver(0, 0, 0, *psi_WO, color="b")

# states labels
offset = 1.2
ax.text(*(offset * beta), r"$\left|\beta\right\rangle$", color="r", fontsize=15)
ax.text(*(offset * alpha), r"$\left|\alpha\right\rangle$", color="g", fontsize=15)
ax.text(*(offset * psi), r"$\left|\psi\right\rangle$", color="b", fontsize=15)
ax.text(*(offset * gamma), r"$\left|\gamma\right\rangle$", color="k", fontsize=15)
ax.text(*(offset * psi_), r"$\left|\psi^{-}\right\rangle$", color="b", fontsize=15)
ax.text(*(offset * psi_O), r"$O\left|\psi\right\rangle$", color="b", fontsize=15)
ax.text(*(offset * psi_WO), r"$WO\left|\psi\right\rangle$", color="b", fontsize=15)

# show
ax.set_xlim([-0.7, 0.7])
ax.set_ylim([-0.7, 0.7])
ax.set_zlim([-0.7, 0.7])
ax.view_init(elev=25, azim=15)
plt.show()
