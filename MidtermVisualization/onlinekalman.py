import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from pykalman import KalmanFilter
import math

class MultiOnlineKalman:
    def __init__(self):
        self.filter_list = []
    
    def take_multiple_observations(self, observations):
        taken_filter_indices = []
        corrected_results = []

        for observation in observations:
            matching_filter_index = self.find_matching_filter_index(observation, taken_filter_indices)
            if matching_filter_index is None:
                new_filter = OnlineKalman()
                corrected_state, _ = new_filter.take_observation(observation[0], observation[1], observation[2])
                self.filter_list.append(new_filter)
                corrected_results.append([corrected_state[0], corrected_state[2], corrected_state[4]])
            else:
                corrected_state, _ = self.filter_list[matching_filter_index].take_observation(observation[0], observation[1], observation[2])
                corrected_results.append([corrected_state[0], corrected_state[2], corrected_state[4]])
        
        return corrected_results
    
    def find_matching_filter_index(self, observation, taken_filter_indices, distance_cap=10):
        closest_index = None
        closest_dist = math.inf

        for idx, some_filter in enumerate(self.filter_list):
            if idx in taken_filter_indices:
                continue
            filter_position = some_filter.get_last_position()
            distance = math.sqrt((filter_position[0]-observation[0])**2 + (filter_position[1]-observation[1])**2 + (filter_position[2]-observation[2])**2)
            if distance < closest_dist and distance < distance_cap:
                closest_index = idx
                closest_dist = distance
        
        if closest_index is not None:
            taken_filter_indices.append(closest_index)

        print("Closest dist: {}".format(closest_dist))
        
        return closest_index

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
    
    def get_last_position(self):
        last_state = self.filtered_state_means[-1]
        return (last_state[0], last_state[2], last_state[4])

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
