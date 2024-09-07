import matplotlib.pyplot as plt
from qutip import Bloch, basis

bloch = Bloch()
bloch.add_states(basis(2, 0) + basis(2, 1))  # 在布洛赫球上绘制基态 |0>
bloch.show()
plt.show()
