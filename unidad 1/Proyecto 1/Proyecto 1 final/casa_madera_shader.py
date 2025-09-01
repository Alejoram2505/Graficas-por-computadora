import pygame

class CRTGlitchShader:
    def __init__(self, base_color=(0, 255, 255)):
        self.base_color = base_color

    def draw_wireframe(self, screen, pts):
        glow_levels = [
            (6, 20),   # grosor y opacidad baja
            (4, 60),
            (2, 120),
            (1, 255),  # centro brillante
        ]
        for width, alpha in glow_levels:
            surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            color = (*self.base_color, alpha)
            pygame.draw.polygon(surface, color, pts, width)
            screen.blit(surface, (0, 0))

    # Este método es ignorado por draw_obj, pero lo necesitamos para cumplir interfaz
    def vertex_shader(self, *args, **kwargs):
        return (0, 255, 255)
