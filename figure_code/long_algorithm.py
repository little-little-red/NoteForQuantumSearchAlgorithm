import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

# LaTeX
plt.rc("text", usetex=True)

# constants
theta = np.pi / 6
j = np.ceil((np.pi / 2 - theta / 2) / theta)
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
I_W = basictrans * qt.Qobj([[1, 0], [0, np.exp(-1j * phi)]]) * basictrans_inv
Q = I_W * I_O

# iterational states
psi_O = I_O * psi
psi_Q = I_W * psi_O
psi_QQ = Q * Q * psi


# Bloch transform
def bloch_vector(state):
    bloch = qt.Bloch()
    bloch.add_states(state)
    return bloch.vectors[-1]


# Initialize the Bloch sphere
bloch = qt.Bloch()

# add states
bloch.add_states(beta)
bloch.add_states(alpha)
bloch.add_states(psi)
bloch.add_states(psi_O)
bloch.add_states(psi_Q)
bloch.add_states(psi_QQ)

# labels
bloch.zlabel = [r"$\left|\alpha\right\rangle$", r"$\left|\beta\right\rangle$"]
bloch.add_annotation(bloch_vector(psi), r"$\left|\psi\right\rangle$")
bloch.add_annotation(bloch_vector(psi_O), r"$I_{O}\left|\psi\right\rangle$")
bloch.add_annotation(bloch_vector(psi_Q), r"$Q\left|\psi\right\rangle$")
bloch.add_annotation(bloch_vector(psi_QQ), r"$Q^{2}\left|\psi\right\rangle$")

# lines


# show
bloch.show()
plt.show()
