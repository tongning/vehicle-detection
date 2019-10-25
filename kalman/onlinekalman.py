import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from pykalman import KalmanFilter

class OnlineKalman:
    def __init__(self):
        self.kalman_filter = None
        self.filtered_state_means = []
        self.filtered_state_covariances = []
        # Encode the model:
        # x(k) = x(k-1) + dt*x_dot(k-1)
        # x_dot(k) = x_dot(k-1)
        # ...same for y and z
        self.transition_matrix =   [[1, 1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 1],
                                    [0, 0, 0, 0, 0, 1]]
        # Our x,y,z observations represent the first, third, and fifth columns of the state.
        self.observation_matrix =  [[1, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0]]

        # transition_covariance 
        self.Q =   [[1e-4,     0,     0,     0,    0,    0], 
                    [   0,  1e-4,     0,     0,    0,    0],
                    [   0,     0,  1e-4,     0,    0,    0],
                    [   0,     0,     0,  1e-4,    0,    0],
                    [   0,     0,     0,     0, 1e-4,    0],
                    [   0,     0,     0,     0,    0, 1e-4]]
        
        self.initial_state_covariance =[[5,    0,   0,    0,    0,   0],
                                        [0,    50,  0,    0,    0,   0],
                                        [0,    0,   5,    0,    0,   0],
                                        [0,    0,   0,    50,   0,   0],
                                        [0,    0,   0,    0,    5,   0],
                                        [0,    0,   0,    0,    0,   50]]
    
    def take_observation(self, x, y, z):
        if self.kalman_filter is None:
            initial_state_mean = [x, 0, y, 0, z, 0]
            # Initialize the Kalman filter
            self.kalman_filter = KalmanFilter(  transition_matrices = self.transition_matrix,
                                                observation_matrices = self.observation_matrix,
                                                initial_state_mean = initial_state_mean,
                                                initial_state_covariance = self.initial_state_covariance,
                                                transition_covariance = self.Q)
            self.filtered_state_means.append(initial_state_mean)
            self.filtered_state_covariances.append(self.initial_state_covariance)
            return (initial_state_mean, self.initial_state_covariance)
        else:
            if x is not None and y is not None and z is not None:
                new_mean, new_cov = (
                self.kalman_filter.filter_update(
                    self.filtered_state_means[-1],
                    self.filtered_state_covariances[-1],
                    observation = [x, y, z])
                )
            else:
                new_mean, new_cov = (
                self.kalman_filter.filter_update(
                    self.filtered_state_means[-1],
                    self.filtered_state_covariances[-1])
                )
            self.filtered_state_means.append(new_mean.tolist())
            self.filtered_state_covariances.append(new_cov)
            return (new_mean.tolist(), new_cov)
