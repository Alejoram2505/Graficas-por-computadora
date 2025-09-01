import pygame
import numpy as np
import random
import math

WIDTH, HEIGHT = 800, 800
BLACK = (0, 0, 0)

# -------------------- Static Noise Shader --------------------
class StaticNoiseShader:
    def __init__(self):
        self.base_color = (101, 67, 33)  
        self.noise_intensity = 0.3

    def vertex_shader(self, position, x, y):
        noise = random.random() * self.noise_intensity
        color = np.array(self.base_color) * (1 - noise) + np.array([255, 255, 255]) * noise
        return tuple(color.astype(int))

# -------------------- Transform Utilities --------------------
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

# -------------------- OBJ Loader --------------------
class Obj:
    def __init__(self, filename):
        self.vertices = []
        self.faces = []
        self.load_model(filename)

    def load_model(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    _, x, y, z = line.strip().split()[:4]
                    self.vertices.append([float(x), float(y), float(z), 1])
                elif line.startswith('f '):
                    parts = line.strip().split()[1:]
                    face = [int(part.split('/')[0]) - 1 for part in parts]
                    if len(face) == 3:
                        self.faces.append(face)
                    elif len(face) > 3:
                        for i in range(1, len(face) - 1):
                            self.faces.append([face[0], face[i], face[i+1]])

# -------------------- Renderer --------------------
class Renderer:
    def __init__(self, width, height, shader):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.shader = shader

    def clear(self):
        self.screen.fill(BLACK)

    def draw_filled_triangle(self, pts):
        color = self.shader.vertex_shader(pts[0], pts[0][0], pts[0][1])
        pygame.draw.polygon(self.screen, color, pts)

    def render(self, model, transform):
        self.clear()
        for face in model.faces:
            pts = []
            for idx in face:
                v = model.vertices[idx]
                v_t = transform @ np.array(v)
                if v_t[3] != 0:
                    v_t /= v_t[3]
                pts.append((int(v_t[0]), int(v_t[1])))
            if len(pts) == 3:
                self.draw_filled_triangle(pts)
        pygame.display.flip()

# -------------------- Main --------------------
def main():
    shader = StaticNoiseShader()
    renderer = Renderer(WIDTH, HEIGHT, shader)
    model = Obj("objetos/banca.obj") 

    # Cámara
    eye = [0, 1, 5]
    target = [0, 0, 0]
    up = [0, 1, 0]

    projection = perspective_projection(45, WIDTH / HEIGHT, 0.1, 100)
    view = look_at(eye, target, up)
    viewport = viewport_matrix(0, 0, WIDTH, HEIGHT)

    # Escala segura (ajustada para que banca se vea)
    model_matrix = translate(0, -1.5, 0) @ scale(0.015, 0.015, 0.015)
    transform = viewport @ projection @ view @ model_matrix

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        renderer.render(model, transform)
        pygame.time.wait(10)

    pygame.quit()

if __name__ == "__main__":
    main()
