from model import NagelSchreckenberg
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

density_range = np.arange(0.0, 1.0, 0.05) ** 2
p_range = np.arange(0, 1.2, 0.2)

for p in tqdm(p_range):
    runvel = []
    runs = 250
    for density in tqdm(density_range):
        model = NagelSchreckenberg(100, 100, 5, p, density)
        model.run(100)
        runvel.append(model.get_mean_velocity())
    plt.plot(density_range, runvel)

plt.title("Nagel–Schreckenberg model")
plt.ylabel("Steady-state velocity")
plt.xlabel("Density")
plt.legend(['p = {0:.1f}'.format(p) for p in p_range])
plt.show()