import pygame
import numpy as np
import math

WIDTH, HEIGHT = 800, 800
BLACK = (0, 0, 0)
CYAN = (0, 255, 255)

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def look_at(eye, target, up):
    forward = normalize(np.array(target) - np.array(eye))
    right = normalize(np.cross(forward, up))
    new_up = normalize(np.cross(right, forward))
    return np.array([
        [right[0], right[1], right[2], -np.dot(right, eye)],
        [new_up[0], new_up[1], new_up[2], -np.dot(new_up, eye)],
        [-forward[0], -forward[1], -forward[2], np.dot(forward, eye)],
        [0, 0, 0, 1]
    ])

def perspective_projection(fov, aspect, near, far):
    f = 1 / math.tan(math.radians(fov) / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ])

def viewport_matrix(x, y, width, height):
    return np.array([
        [width/2, 0, 0, x + width/2],
        [0, -height/2, 0, y + height/2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def translate(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

def scale(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

class Obj:
    def __init__(self, filename):
        self.vertices = []
        self.edges = set()
        self.load_model(filename)

    def load_model(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    _, x, y, z = line.split()
                    self.vertices.append([float(x), float(y), float(z), 1])
                elif line.startswith('f '):
                    indices = [int(part.split('/')[0]) - 1 for part in line.strip().split()[1:]]
                    for i in range(len(indices)):
                        self.edges.add((indices[i], indices[(i+1)%len(indices)]))

class NeonShader:
    def __init__(self, base_color=(0, 255, 255)):
        self.base_color = base_color

    def draw_edge(self, screen, p1, p2):
        glow_levels = [
            (6, 20),   # grosor, opacidad baja
            (4, 60),
            (2, 120),
            (1, 255),  # centro brillante
        ]
        for width, alpha in glow_levels:
            surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            color = (*self.base_color, alpha)
            pygame.draw.line(surface, color, p1, p2, width)
            screen.blit(surface, (0, 0))

class Renderer:
    def __init__(self, width, height):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.shader = NeonShader()

    def clear(self):
        self.screen.fill(BLACK)

    def render_wireframe(self, model, transform):
        self.clear()
        projected = []
        for v in model.vertices:
            v_t = transform @ np.array(v)
            if v_t[3] != 0:
                v_t /= v_t[3]
            projected.append((int(v_t[0]), int(v_t[1])))

        for idx1, idx2 in model.edges:
            p1 = projected[idx1]
            p2 = projected[idx2]
            self.shader.draw_edge(self.screen, p1, p2)

        pygame.display.flip()

def main():
    renderer = Renderer(WIDTH, HEIGHT)
    model = Obj("objetos/casa_madera.obj")

    eye = [0, 1, 5]
    target = [0, 0, 0]
    up = [0, 1, 0]

    projection = perspective_projection(45, WIDTH / HEIGHT, 0.1, 100)
    view = look_at(eye, target, up)
    viewport = viewport_matrix(0, 0, WIDTH, HEIGHT)
    model_matrix = translate(0, -0.5, 0) @ scale(0.5, 0.5, 0.5)
    transform = viewport @ projection @ view @ model_matrix

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        renderer.render_wireframe(model, transform)
        pygame.time.wait(10)

    pygame.quit()

if __name__ == "__main__":
    main()
