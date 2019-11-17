import numpy as np
import random
from .Position import Position


class Particle():
    def __init__(self, data, num_particle, index):
        total_data = len(data)
        self.data = data[total_data/num_particle*index:total_data/num_particle*(index+1)-1]
        self.position = Position()
        self.pbest_position = self.position
        self.pbest_value = 1000
        self.velocity = [0,0,0,0]

    def __str__(self):
        print("I am at ", self.position, " meu pbest is ", self.pbest_position)

    def move(self):
        self.position.learning_rate_rbm = self.position.learning_rate_rbm + self.velocity[0]
        self.position.learning_rate = self.position.learning_rate + self.velocity[1]
        self.position.number_visible_input = self.position.number_visible_input + self.velocity[2]
        self.position.number_hidden_input = self.position.number_hidden_input + self.velocity[3]
