import random
import numpy as np
from Particle.Position import Position
from HSMemory import HSMemory
import copy
from TensorGlobal import TensorGlobal

class Space:

    W = 0.5
    c1 = 0.8
    c2 = 0.9

    def __init__(self, target, target_error, n_particles):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = 1000
        self.gbest_position = Position(0, 0)

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()

    def fitness(self, particle):
        tmpPosition = copy.deepcopy(particle.position)
        tmpPosition = HSMemory. \
        update_mse(tmpPosition, particle.data)
        return tmpPosition
        
    def set_pbest_gbest(self):
        loop_index = 1
        for particle in self.particles:
            fitness_cadidate = self.fitness(particle)
            #insert more infor for result
            tmp_last_result = TensorGlobal.followHs[-1]
            tmp_last_result[6] = "Particle_" + str(loop_index)
            if(particle.pbest_value > fitness_cadidate.test_mse):
                particle.pbest_value = fitness_cadidate.test_mse
                particle.pbest_position = particle.position
                tmp_last_result[7] = fitness_cadidate.test_mse
            # set gbest
            if(self.gbest_value > fitness_cadidate.test_mse):
                self.gbest_value = fitness_cadidate.test_mse
                self.gbest_position = particle.position
                tmp_last_result[8] = fitness_cadidate.test_mse
            loop_index = loop_index + 1

    #def set_gbest(self):
        #for particle in self.particles:
            #best_fitness_cadidate = self.fitness(particle)
            #if(self.gbest_value > best_fitness_cadidate):
                #self.gbest_value = best_fitness_cadidate
                #self.gbest_position = particle.position

    def move_particles(self):
        for particle in self.particles:
            lrr_velocity = (Space.W*particle.velocity[0]) + (Space.c1*random.random()) * (particle.pbest_position.learning_rate_rbm - particle.position.learning_rate_rbm) + \
                           (random.random()*Space.c2) * (self.gbest_position.learning_rate_rbm - particle.position.learning_rate_rbm)
            lr_velocity = (Space.W*particle.velocity[1]) + (Space.c1*random.random()) * (particle.pbest_position.learning_rate - particle.position.learning_rate) + \
                           (random.random()*Space.c2) * (self.gbest_position.learning_rate - particle.position.learning_rate)
            number_visible_input_velocity = (Space.W*particle.velocity[1]) + (Space.c1*random.random()) * (particle.pbest_position.number_visible_input - particle.position.number_visible_input) + \
                           (random.random()*Space.c2) * (self.gbest_position.number_visible_input - particle.position.number_visible_input)
            number_hidden_input_velocity = (Space.W*particle.velocity[1]) + (Space.c1*random.random()) * (particle.pbest_position.number_hidden_input - particle.position.number_hidden_input) + \
                           (random.random()*Space.c2) * (self.gbest_position.number_hidden_input - particle.position.number_hidden_input)
            particle.velocity = [lrr_velocity, lr_velocity, number_visible_input_velocity, number_hidden_input_velocity]
            particle.move()
