import numpy as np

class NeonShader:
    def __init__(self):
        self.base_color = np.array([0, 255, 255])  # cian brillante tipo neón

    def vertex_shader(self, position, normal, light_dir, view_dir, x, y):
        glow = (np.sin(x * 0.05) + 1) * 0.5  # efecto de pulso
        color = self.base_color * glow
        color = np.clip(color, 0, 255)
        return tuple(color.astype(int))
