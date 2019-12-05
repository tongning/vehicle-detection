import numpy as np
from scipy.stats import multivariate_normal
import random
from typing import List, Any
from statistics import mean


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
    def __init__(self):
        self.filtered_positions = []
        self.particle_locations = []
        self.particle_weights = []
        self.initial_particle_radius = 20
        self.num_particles = 100

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

    def take_observation(self, x, y, z):
        print(len(self.particle_locations))
        if len(self.filtered_positions) == 1:
            self.generate_random_particles([x, y, z], self.initial_particle_radius)

        if len(self.filtered_positions) <= 1:
            self.filtered_positions.append([x, y, z])
            return [x, y, z]
        
        # Move all of the particles based on velocity
        new_particles = []
        new_weights = []

        prev_location = self.get_last_position()
        velocity = [x - prev_location[0], y - prev_location[1], z - prev_location[2]]
        for prev_particle in self.particle_locations:
            particle_new_location = [prev_particle[0]+velocity[0], prev_particle[1]+velocity[1], prev_particle[2]-velocity[2]]
            new_particles.append(particle_new_location)
            new_weights.append(get_3d_gaussian_pdf([x, y, z], 20, particle_new_location))

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
        


        

            

        
        
        

    