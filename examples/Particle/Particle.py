#  import numpy as np
#  import random
from Particle.Position import Position


class Particle():
    def __init__(self, data, num_particle, index):
        self.data = data
        self.position = Position(num_particle, index)
        self.pbest_position = self.position
        self.pbest_value = 1000
        self.velocity = [0,0,0,0]

    def __str__(self):
        print("I am at ", self.position, " meu pbest is ", self.pbest_position)

    def move(self):
        self.position.learning_rate_rbm = self.position.learning_rate_rbm + self.velocity[0]
        self.position.learning_rate = self.position.learning_rate + self.velocity[1]
        self.position.number_visible_input = int(self.position.number_visible_input + self.velocity[2])
        self.position.number_hidden_input = int(self.position.number_hidden_input + self.velocity[3])

        if self.position.learning_rate_rbm > self.position.config_lrr_max:
            self.position.learning_rate_rbm = self.position.config_lrr_max

        if self.position.learning_rate_rbm < self.position.config_lrr_min:
            self.position.learning_rate_rbm = self.position.config_lrr_min

        if self.position.learning_rate > self.position.config_lr_max:
            self.position.learning_rate = self.position.config_lr_max

        if self.position.learning_rate < self.position.config_lr_min:
            self.position.learning_rate = self.position.config_lr_min

        if self.position.number_visible_input > self.position.config_number_visible_input_max:
            self.position.number_visible_input = self.position.config_number_visible_input_max

        if self.position.number_visible_input < self.position.config_number_visible_input_min:
            self.position.number_visible_input = self.position.config_number_visible_input_min

        if self.position.number_hidden_input > self.position.config_number_hidden_input_max:
            self.position.number_hidden_input = self.position.config_number_hidden_input_max

        if self.position.number_hidden_input < self.position.config_number_hidden_input_min:
            self.position.number_hidden_input = self.position.config_number_hidden_input_min
