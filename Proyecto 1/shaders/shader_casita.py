import pygame
import numpy as np
import random

WIDTH = 1000
HEIGHT = 700

class CRTGlitchShader:
    def __init__(self):
        self.base_color = np.array([180, 180, 255], dtype=np.float32)

    def vertex_shader(self, frag_pos, normal, light_dir, view_dir, x, y):
        base = self.base_color.copy()

        if (y % 12) < 6:
            base *= 0.2

        noise_strength = random.uniform(0.4, 1.8)
        base *= noise_strength

        dx = (x - WIDTH / 2) / (WIDTH / 2)
        dy = (y - HEIGHT / 2) / (HEIGHT / 2)
        distortion = 1 + 0.4 * (dx**2 + dy**2)
        base *= 1 / distortion

        glitch_scale = 50
        shift_r = random.randint(-3, 3)
        shift_g = random.randint(-3, 3)
        shift_b = random.randint(-3, 3)

        r = np.clip(base[0] + shift_r * glitch_scale, 0, 255)
        g = np.clip(base[1] + shift_g * glitch_scale, 0, 255)
        b = np.clip(base[2] + shift_b * glitch_scale, 0, 255)

        return np.array([r, g, b], dtype=np.uint8)
