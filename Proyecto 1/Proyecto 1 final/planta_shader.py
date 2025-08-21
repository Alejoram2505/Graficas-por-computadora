import pygame
import numpy as np
import os

WIDTH, HEIGHT = 1000, 700
BLACK = (0, 0, 0)

# -------------------- Shader de gradiente vertical --------------------
class GradientShader:
    def __init__(self):
        self.color1 = np.array([0, 255, 100]) 
        self.color2 = np.array([0, 100, 50])   

    def vertex_shader(self, position, x, y, height):
        t = (y % height) / height if height != 0 else 0
        color = self.color1 * (1 - t) + self.color2 * t
        return color.astype(int)

# -------------------- OBJ Loader --------------------
class OBJ:
    def __init__(self, path):
        self.vertices = []
        self.faces_v = []
        self._load(path)

    @staticmethod
    def _f(x):
        try:
            return float(x)
        except:
            return 0.0

    def _load(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith("v "):
                    _, x, y, z = (line + " 0 0").split()[:4]
                    self.vertices.append((self._f(x), self._f(y), self._f(z)))
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    v_idx = []
                    for t in parts:
                        v = int(t.split('/')[0]) - 1
                        v_idx.append(v)
                    for i in range(1, len(v_idx) - 1):
                        self.faces_v.append([v_idx[0], v_idx[i], v_idx[i+1]])

# -------------------- Helpers --------------------
def project_point(v, pos, scale):
    x, y, z = v
    perspective_factor = 1.0 + z * 0.001
    sx = int(x * scale * perspective_factor + pos[0])
    sy = int(-y * scale * perspective_factor + pos[1])
    return (sx, sy)

def draw_obj_gradient(screen, obj, pos, scale, shader):
    for tri in obj.faces_v:
        pts = [project_point(obj.vertices[i], pos, scale) for i in tri]
        if len(pts) == 3:
            min_y = min(p[1] for p in pts)
            max_y = max(p[1] for p in pts)
            height = max_y - min_y
            center_y = (min_y + max_y) // 2
            color = shader.vertex_shader((0, center_y), pts[0][0], center_y, height)
            pygame.draw.polygon(screen, color, pts)

# -------------------- Main --------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Planta con Shader de Gradiente")

    # Ruta al modelo
    obj_path = os.path.join("objetos", "Planta.obj")
    planta = OBJ(obj_path)

    shader = GradientShader()

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        position = (WIDTH // 2, HEIGHT // 2 + 80)  
        scale = 60  

        draw_obj_gradient(screen, planta, position, scale, shader)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
