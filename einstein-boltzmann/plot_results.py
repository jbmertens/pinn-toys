import numpy as np
import matplotlib.pyplot as plt

base_dir = "outputs/meszaros_solver/validators/"

# plot in 1d
data = np.load(base_dir + "validator.npz", allow_pickle=True)
data = np.atleast_1d(data.f.arr_0)[0]

plt.loglog(data["y"], data["true_delta_m"], label="True delta_m")
plt.loglog(data["y"], data["pred_delta_m"], label="Pred delta_m")
plt.legend()
print(data["true_delta_m"])
print(data["pred_delta_m"])
plt.savefig("comparison.png")

'''
fig, ax = plt.subplots(2,1, sharex=True)
fig.subplots_adjust(hspace=0.001)

ax[0].loglog(data["x"], data["true_y"], label="True y")
ax[0].loglog(data["x"], data["pred_y"], label="Pred y")
#print(data["pred_y"]-data["true_y"])
ax[1].loglog(data["x"], np.abs(data["pred_y"]-data["true_y"]))

print(data["pred_y"], data["true_y"])
fig.legend()
fig.savefig("comparison.png")
'''

