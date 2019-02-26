from model import NagelSchreckenberg
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

density_range = np.arange(0.1, 0.4, 0.1)
p_range = np.arange(0, 0.2, 0.2)

max_vel = 20

for density in tqdm(density_range):
    runvel = []
    runs = 250
    # for density in tqdm(density_range):
    model = NagelSchreckenberg(100, 300, max_vel, 0.00, density)
    model.run(100)
    runvel.append(model.get_mean_velocity())
    # plt.plot(density_range, runvel)
    plt.plot(np.arange(max_vel + 1), model.get_velocity_histogram())

plt.title("Nagel–Schreckenberg model")
plt.ylabel("probability")
plt.xlabel("velocity")
plt.legend(['density = {0:.1f}'.format(d) for d in density_range])
plt.show()

