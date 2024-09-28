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

# operation
Oracle = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
basictrans_A = np.array(
    [
        [1, 0, 0],
        [0, np.cos(theta / 2), -np.sin(theta / 2)],
        [0, np.sin(theta / 2), np.cos(theta / 2)],
    ]
)
basictrans_A_inv = np.linalg.inv(basictrans_A)
cD_diag = np.array([-1, 1, -1])
cD = np.dot(np.dot(basictrans_A, np.diag(cD_diag)), basictrans_A_inv)
basictrans_B = np.array(
    [
        [np.cos(omega / 2), 0, -np.sin(omega / 2)],
        [0, 1, 0],
        [np.sin(omega / 2), 0, np.cos(omega / 2)],
    ]
)
basictrans_B_inv = np.linalg.inv(basictrans_B)
basictrans = np.dot(basictrans_A, basictrans_B)
basictrans_inv = np.linalg.inv(basictrans)

# states
beta = np.array([0, 0, 1])
alpha = np.array([0, 1, 0])
gamma = np.array([1, 0, 0])
psi = np.dot(basictrans_A, alpha)
psi_perp = np.dot(basictrans_A, beta)
psi_ = np.dot(basictrans_B, gamma)
tau = np.dot(basictrans_B, psi_perp)

# iterational states
psi_O = np.dot(Oracle, psi)
psi_G = np.dot(cD, psi_O)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# add states
ax.quiver(0, 0, 0, *beta, color="k")
ax.quiver(0, 0, 0, *alpha, color="k")
ax.quiver(0, 0, 0, *gamma, color="k")
ax.quiver(0, 0, 0, *psi, color="r")
ax.quiver(0, 0, 0, *psi_perp, color="b")
ax.quiver(0, 0, 0, *psi_, color="r")
ax.quiver(0, 0, 0, *tau, color="r")

# states labels
offset = 1.2
ax.text(*(offset * beta), r"$\left|\beta\right\rangle$", color="r", fontsize=15)
ax.text(*(offset * alpha), r"$\left|\alpha\right\rangle$", color="k", fontsize=15)
ax.text(*(offset * psi), r"$\left|\psi\right\rangle$", color="k", fontsize=15)
ax.text(*(offset * gamma), r"$\left|\gamma\right\rangle$", color="k", fontsize=15)
ax.text(
    *(offset * psi_perp), r"$\left|\psi_{\perp}\right\rangle$", color="k", fontsize=15
)
ax.text(*(offset * psi_), r"$\left|\psi^{-}\right\rangle$", color="k", fontsize=15)
ax.text(*(offset * tau), r"$\left|\tau\right\rangle$", color="k", fontsize=15)


# show
ax.set_xlim([-0.7, 0.7])
ax.set_ylim([-0.7, 0.7])
ax.set_zlim([-0.7, 0.7])
ax.view_init(elev=25, azim=15)
plt.show()
