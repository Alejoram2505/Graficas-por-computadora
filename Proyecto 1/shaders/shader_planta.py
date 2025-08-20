import pygame
import numpy as np

class GradientShader:
    def __init__(self):
        self.color1 = np.array([0, 255, 100])  
        self.color2 = np.array([0, 100, 50])   

    def vertex_shader(self, position, x, y, height):
        # Calcula el gradiente basado en la posición vertical
        t = (y % height) / height if height != 0 else 0
        color = self.color1 * (1 - t) + self.color2 * t
        return color.astype(int)
