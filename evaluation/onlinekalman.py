import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from pykalman import KalmanFilter
import math

class MultiOnlineKalman:
    def __init__(self, sequence_name):
        self.filter_list = []
        self.sequence_name = sequence_name

    def distance(self, pos1, pos2):
        return math.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2+(pos1[2]-pos2[2])**2)

    def take_multiple_observations(self, observations, detection_confidences):

        taken_filter_set = set()
        corrected_results = []
        corrected_confidences = []

        for idx, observation in enumerate(observations):
            matching_filter_index = self.find_matching_filter_index(observation, taken_filter_set)
            if matching_filter_index is None:
                new_filter = OnlineKalman()
                corrected_state, _ = new_filter.take_observation(observation[0], observation[1], observation[2])
                new_filter.detection_confidence = detection_confidences[idx]
                self.filter_list.append(new_filter)
                taken_filter_set.add(new_filter)
                corrected_results.append(observation)
                corrected_confidences.append(detection_confidences[idx])
            else:
                taken_filter_set.add(self.filter_list[matching_filter_index])
                corrected_state, _ = self.filter_list[matching_filter_index].take_observation(observation[0], observation[1], observation[2])
                self.filter_list[matching_filter_index].detection_confidence = detection_confidences[idx]
                if False:#self.distance([corrected_state[0], corrected_state[2], corrected_state[4]], observation) > 4:
                    corrected_results.append(observation)
                else:
                    corrected_results.append([corrected_state[0], corrected_state[2], corrected_state[4]])
                corrected_confidences.append(detection_confidences[idx])

        print("Percentage of filters matched: {}; {}/{}".format(len(taken_filter_set)/len(self.filter_list),len(taken_filter_set), len(self.filter_list)))



        filters_to_remove = set()
        for idx, filt in enumerate(self.filter_list):
            if filt not in taken_filter_set:
                filt.detection_confidence -= 0.05
                if filt.detection_confidence < 0:
                    filt.detection_confidence = 0
                    filters_to_remove.add(filt)

            if filt.time_since_last_update > 10:
                filters_to_remove.add(filt)
        for filt in filters_to_remove:
            self.filter_list.remove(filt)


        for idx, filt in enumerate(self.filter_list):
            if filt not in taken_filter_set:
                corrected_state, _ = self.filter_list[idx].take_observation(None, None, None)
                corrected_results.append([corrected_state[0], corrected_state[2], corrected_state[4]])
                corrected_confidences.append(filt.detection_confidence)


        return corrected_results, corrected_confidences

    def find_matching_filter_index(self, observation, taken_filter_set, distance_cap=8):
        closest_index = None
        closest_dist = math.inf

        for idx, some_filter in enumerate(self.filter_list):
            if idx in taken_filter_set:
                continue
            filter_position = some_filter.get_last_position()
            distance = math.sqrt((filter_position[0]-observation[0])**2 + (filter_position[1]-observation[1])**2 + (filter_position[2]-observation[2])**2)
            if distance < closest_dist and distance < distance_cap:
                closest_index = idx
                closest_dist = distance

        #print("Closest dist: {}".format(closest_dist))

        return closest_index

class OnlineKalman:
    def __init__(self):
        self.kalman_filter = None
        self.detection_confidence = None
        self.filtered_state_means = []
        self.filtered_state_covariances = []
        self.time_since_last_update = 0
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

        self.initial_state_covariance =[[1,    0,   0,    0,    0,   0],
                                        [0,    50,  0,    0,    0,   0],
                                        [0,    0,   1,    0,    0,   0],
                                        [0,    0,   0,    50,   0,   0],
                                        [0,    0,   0,    0,    1,   0],
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
                                                initial_state_covariance = self.initial_state_covariance
                                                #transition_covariance = self.Q
                                                )
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
                self.time_since_last_update += 1
                new_mean, new_cov = (
                self.kalman_filter.filter_update(
                    self.filtered_state_means[-1],
                    self.filtered_state_covariances[-1])
                )
                #print("Got here {}".format(self.get_last_position()))
            self.filtered_state_means.append(new_mean.tolist())
            self.filtered_state_covariances.append(new_cov)
            return (new_mean.tolist(), new_cov)
