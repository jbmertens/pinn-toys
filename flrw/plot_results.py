import numpy as np
import matplotlib.pyplot as plt

base_dir = "outputs/flrw_solver/validators/"

# plot in 1d
data = np.load(base_dir + "validator.npz", allow_pickle=True)
data = np.atleast_1d(data.f.arr_0)[0]

plt.loglog(data["t"], data["true_a"], label="True a")
plt.loglog(data["t"], data["true_H"], label="True H")
plt.loglog(data["t"], data["pred_a"], label="Pred a")
plt.loglog(data["t"], data["pred_H"], label="Pred H")

plt.legend()
plt.savefig("comparison.png")
