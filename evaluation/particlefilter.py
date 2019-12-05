import numpy as np
from scipy.stats import multivariate_normal
import random
from typing import List, Any
from statistics import mean
import math


class MultiOnlineParticleFilter:
    def __init__(self, sequence_name):

        # Tunables
        self.num_frames_to_keep_stale_filters = 3
        self.filter_match_distance_cap = 3
        self.num_particles = 100
        self.initial_particle_radius=1
        self.gaussian_stdev=9
        # End of tunables
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
                print("Found no matching filter for {}".format(observation))
                new_filter = ParticleFilter(num_particles=self.num_particles, initial_particle_radius=self.initial_particle_radius, gaussian_stdev=self.gaussian_stdev)
                corrected_state = new_filter.take_observation(observation[0], observation[1], observation[2])

                new_filter.detection_confidence = detection_confidences[idx]

                self.filter_list.append(new_filter)
                taken_filter_set.add(new_filter)

                if not None in list(observation):
                    corrected_results.append(list(observation))
                    corrected_confidences.append(detection_confidences[idx])
            else:
                taken_filter_set.add(self.filter_list[matching_filter_index])
                corrected_state = self.filter_list[matching_filter_index].take_observation(observation[0], observation[1], observation[2])
                self.filter_list[matching_filter_index].detection_confidence = detection_confidences[idx]
                if False and not None in list(observation):#self.distance([corrected_state[0], corrected_state[2], corrected_state[4]], observation) > 4:
                    corrected_results.append(list(observation))
                    corrected_confidences.append(detection_confidences[idx])
                elif not None in list(corrected_state):
                    corrected_results.append(list(corrected_state))
                    corrected_confidences.append(detection_confidences[idx])
                

        #print("Percentage of filters matched: {}; {}/{}".format(len(taken_filter_set)/len(self.filter_list),len(taken_filter_set), len(self.filter_list)))



        filters_to_remove = set()
        for idx, filt in enumerate(self.filter_list):
            if filt not in taken_filter_set:
                filt.detection_confidence -= 0.00
                if filt.detection_confidence < 0:
                    filt.detection_confidence = 0
                    filters_to_remove.add(filt)

            if filt.time_since_last_update > self.num_frames_to_keep_stale_filters:
                filters_to_remove.add(filt)
        for filt in filters_to_remove:
            self.filter_list.remove(filt)


        for idx, filt in enumerate(self.filter_list):
            if filt not in taken_filter_set:
                corrected_state = self.filter_list[idx].take_observation(None, None, None)
                if not None in list(corrected_state):
                    corrected_results.append(list(corrected_state))
                    corrected_confidences.append(0.01)


        return corrected_results, corrected_confidences

    def find_matching_filter_index(self, observation, taken_filter_set):
        distance_cap = self.filter_match_distance_cap
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


def weighted_sample(choices: List[Any], probs: List[float]):
    """
    Sample from `choices` with probability according to `probs`
    """
    probs = np.concatenate(([0], np.cumsum(probs)))
    r = random.random()
    for j in range(len(choices) + 1):
        if probs[j] < r <= probs[j + 1]:
            return choices[j]

def get_3d_gaussian_pdf(mean_xyz, sig, test_xyz):


    mu = np.array(mean_xyz)

    sigma = np.array([sig, sig, sig])
    covariance = np.diag(sigma**2)

    z = multivariate_normal.pdf([test_xyz], mean=mu, cov=covariance)
    return z

class ParticleFilter:
    def __init__(self, num_particles=100, initial_particle_radius=1, gaussian_stdev=9):
        self.filtered_positions = []
        self.particle_locations = []
        self.particle_weights = []
        self.initial_particle_radius = initial_particle_radius
        self.time_since_last_update = 0
        self.detection_confidence = None
        self.num_particles = num_particles
        self.gaussian_stdev = gaussian_stdev

    def generate_random_particles(self, center_xyz, radius):
        for _ in range(self.num_particles):
            x_pos = random.uniform(center_xyz[0]-radius, center_xyz[0]+radius)
            y_pos = random.uniform(center_xyz[1]-radius, center_xyz[1]+radius)
            z_pos = random.uniform(center_xyz[2]-radius, center_xyz[2]+radius)
            self.particle_locations.append([x_pos, y_pos, z_pos])
            self.particle_weights.append(1.0/self.num_particles)

    def get_last_position(self):
        if len(self.filtered_positions) == 0:
            return None
        return self.filtered_positions[-1]

    def average_recent_velocity(self, velocity_list, num_recent_frames=3):
        if len(velocity_list) == 0 or len(velocity_list) == 1:
            return [0, 0, 0]

        x_velocities = []
        y_velocities = []
        z_velocities = []

        num_velocities_added = 0
        index = len(velocity_list) - 1
        while index >= 1 and num_velocities_added < num_recent_frames:
            x_velocities.append(velocity_list[index][0]-velocity_list[index-1][0])
            y_velocities.append(velocity_list[index][1]-velocity_list[index-1][1])
            z_velocities.append(velocity_list[index][2]-velocity_list[index-1][2])
            index -= 1
            num_velocities_added += 1

        return [sum(x_velocities)/len(x_velocities),
                sum(y_velocities)/len(y_velocities),
                sum(z_velocities)/len(z_velocities)]

    def take_observation(self, x, y, z):
        #print(len(self.particle_locations))
        if len(self.filtered_positions) <= 1 and (x is None or y is None or z is None):
            self.time_since_last_update += 1
            return [x,y,z]

        if len(self.filtered_positions) == 1:
            self.filtered_positions.append([x, y, z])
            self.generate_random_particles([x, y, z], self.initial_particle_radius)
            return [x, y, z]
        elif len(self.filtered_positions) == 0:
            self.filtered_positions.append([x, y, z])
            return [x, y, z]
        
        # Move all of the particles based on velocity
        new_particles = []
        new_weights = []

        prev_location = self.get_last_position()
        if x is not None and y is not None and z is not None:
            location_history = self.filtered_positions+[[x,y,z]]
        else:
            location_history = self.filtered_positions
            self.time_since_last_update += 1
        velocity = self.average_recent_velocity(location_history)
        for prev_particle in self.particle_locations:
            particle_new_location = [prev_particle[0]+velocity[0], prev_particle[1]+velocity[1], prev_particle[2]+velocity[2]]
            new_particles.append(particle_new_location)
            if x is not None and y is not None and z is not None:
                new_weights.append(get_3d_gaussian_pdf([x, y, z], self.gaussian_stdev, particle_new_location))
            else:
                new_weights.append(get_3d_gaussian_pdf([prev_location[0]+velocity[0], prev_location[1]+velocity[1], prev_location[2]+velocity[2]], self.gaussian_stdev, particle_new_location))


        if sum(new_weights) != 0:
            new_weights_norm = [float(i)/sum(new_weights) for i in new_weights]

            # Sample from new_particles according to new_weights_norm
            self.particle_locations = []
            self.particle_weights = []
            for _ in range(self.num_particles):
                self.particle_locations.append(weighted_sample(new_particles, new_weights_norm))

            particle_locations_np = np.array(self.particle_locations)
            adjusted_position = np.mean(particle_locations_np, axis=0)
            self.filtered_positions.append(list(adjusted_position))

            return adjusted_position
        else:
            # The observation is really far away from all previously observed values. We should just
            # regenerate new particles around the new observation.
            self.particle_locations = []
            self.particle_weights = []
            self.filtered_positions = []
            self.filtered_positions.append([x, y, z])
            self.generate_random_particles([x, y, z], self.initial_particle_radius)
            return [x, y, z]
        


        

            

        
        
        

    