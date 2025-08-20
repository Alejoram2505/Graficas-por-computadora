import numpy as np

class SimpleLineShader:
    def __init__(self):
        self.line_spacing = 4  # Espaciado entre líneas
        self.line_color = (200, 150, 100)  
        self.background_color = (150, 100, 50)  

    def vertex_shader(self, position, x, y):
        # Crear patrón de líneas diagonales
        if ((x + y) // self.line_spacing) % 2 == 0:
            return self.line_color
        return self.background_color



