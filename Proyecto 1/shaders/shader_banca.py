import pygame
import numpy as np
import random

class StaticNoiseShader:
    def __init__(self):
        self.base_color = (101, 67, 33) 
        self.noise_intensity = 0.3  # Intensidad del ruido

    def vertex_shader(self, position, x, y):
        # Genera ruido aleatorio
        noise = random.random() * self.noise_intensity
        color = np.array(self.base_color) * (1 - noise) + np.array([255, 255, 255]) * noise
        return tuple(color.astype(int))
