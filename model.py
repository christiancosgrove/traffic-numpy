# Efficient numpy implementation of Nagel-Schreckenberg traffic model
# Christian Cosgrove, 2018

import numpy as np
class NagelSchreckenberg:

    def __init__(self, T, N, v, p, density):
        self.T = T
        self.N  = N
        self.v = v
        self.p = p
        self.density = density

        self.states = None
        self.velocities = None
        self.meanvel = None

        self.vel_histogram = np.zeros(self.v + 1)

    """ Simultaneously computes `runs` simulations of the Nagel-Schreckenberg model"""
    def run(self, runs, save_states=False):

        # Store the mean velocity of cars at timestep t
        meanvel = []

        # binary vector - 1 if car, 0 otherwise
        state = np.random.binomial(1, self.density, size=(runs, self.N))

        # stores the velocity of each car
        vel = state * np.random.randint(self.v+1, size=(runs, self.N))
        if save_states:
            self.states = []
            self.velocities = []

        for t in range(self.T):
            # Acceleration
            vel += 1
            vel = np.clip(vel, 0, self.v)
            vel[state == 0] = 0

            # Slowing down
            for k in range(self.v):
                vel[np.logical_and(np.roll(state, -k - 1, axis=1) > 0, (vel >= k + 1))] = k

            # Randomization
            vel -= np.random.binomial(1, self.p, size=(runs, self.N))
            vel = np.clip(vel, 0, self.v)

            # Car motion
            new = np.zeros_like(state)
            indices = np.tile(np.arange(state.shape[1]), (runs, 1)) + vel
            indices = indices % state.shape[1]

            # reshape indices
            indices = np.stack([np.tile(np.arange(runs), (self.N, 1)).T, indices], axis=2)
            indices = indices[state > 0, :]
            indices = np.reshape(indices, (-1, 2))

            nvel = vel[state > 0]
            vel[:] = 0
            vel[indices[:,0], indices[:,1]] = nvel
            state[:] = 0
            state[indices[:, 0], indices[:, 1]] = 1

            # We only want the steady-state velocity, so let the system reach equilibrium first
            if t > self.T // 2:
                meanvel.append(np.mean(vel[state > 0]))

            # print(np.histogram(vel, self.v))
            self.vel_histogram += np.histogram(vel[state > 0], self.v + 1)[0]

            if save_states:
                self.states.append(np.array(state))
                self.velocities.append(np.array(vel))

        self.meanvel = np.mean(meanvel)
        if save_states:
            self.states = np.array(self.states)

    def get_mean_velocity(self):
        return self.meanvel

    def get_states(self):
        return self.states, self.velocities

    def get_velocity_histogram(self):
        return self.vel_histogram / np.sum(self.vel_histogram)
