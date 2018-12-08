from model import NagelSchreckenberg
import numpy as np
import matplotlib.pyplot as plt

for k, density in enumerate(np.arange(0.1, 0.8, 0.1)):
    model = NagelSchreckenberg(200, 50, 5, 0.2, density)
    model.run(1, save_states=True)

    plt.subplot(171 + k)
    states, _ = model.get_states()
    plt.imshow(states[:, 0, :])
    plt.axis('off')
    plt.title('density {0:.1f}'.format(density))


plt.show()