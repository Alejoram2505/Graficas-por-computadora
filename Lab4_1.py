import pygame
import numpy as np
import math
import random

# Configuración de la ventana
WIDTH = 800
HEIGHT = 800

# Utils
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def hsv_to_rgb(h, s, v):
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i %= 6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)

# Disco Light
class DiscoLight:
    def __init__(self, radius, speed, phase):
        self.radius = radius
        self.speed = speed
        self.phase = phase
        self.hue_offset = random.random()
        self.blink_phase = random.random()

    def position(self, time):
        angle = time * self.speed + self.phase
        cx = WIDTH // 2
        cy = HEIGHT // 2
        x = int(cx + self.radius * math.cos(angle))
        y = int(cy + self.radius * math.sin(angle))
        return x, y

    def color(self, time):
        hue = (self.hue_offset + time * 0.1) % 1.0
        brightness = (math.sin(time * 3 + self.blink_phase * 2 * math.pi) + 1) / 2  # 0 to 1
        rgb = hsv_to_rgb(hue, 1.0, brightness)
        return tuple(int(c * 255) for c in rgb)

# Shader
class DiscoShader:
    def __init__(self):
        self.time = 0
        self.sky_color = (10, 10, 40)
        self.lights = [DiscoLight(250, 0.3 + 0.1*i, 2 * math.pi * i / 12) for i in range(12)]

    def get_sky_color(self):
        return self.sky_color

    def vertex_shader(self, vertex, normal, light_dir):
        intensity = max(0.2, np.dot(normal, light_dir))
        return intensity, vertex[2]

    def fragment_shader(self, intensity, depth, tex_color, x, y):
        r, g, b = tex_color[:3]
        result_color = np.array([r, g, b], dtype=np.float32)

        for light in self.lights:
            lx, ly = light.position(self.time)
            dist = math.sqrt((x - lx)**2 + (y - ly)**2)
            if dist < 150:
                fade = max(0, 1 - dist / 150)
                lr, lg, lb = light.color(self.time)
                light_color = np.array([lr, lg, lb])
                result_color += light_color * fade * 0.7

        return tuple(np.clip(result_color, 0, 255).astype(np.uint8))

# OBJ loader
class Obj:
    def __init__(self, filename, texture_filename):
        self.vertices = []
        self.tex_coords = []
        self.normals = []
        self.faces = []
        self.load_model(filename)
        self.load_texture(texture_filename)

    def load_model(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    _, x, y, z = line.split()
                    self.vertices.append([float(x), float(y), float(z), 1])
                elif line.startswith('vt '):
                    _, u, v = line.split()
                    self.tex_coords.append([float(u), 1 - float(v)])
                elif line.startswith('vn '):
                    _, nx, ny, nz = line.split()
                    self.normals.append([float(nx), float(ny), float(nz)])
                elif line.startswith('f '):
                    face, tex_face, norm_face = [], [], []
                    parts = line.strip().split()[1:]
                    for part in parts:
                        v, t, n = (part.split('/') + ['0', '0'])[:3]
                        face.append(int(v)-1)
                        tex_face.append(int(t)-1 if t else 0)
                        norm_face.append(int(n)-1 if n else 0)
                    for i in range(1, len(face)-1):
                        self.faces.append({
                            'vertices': [face[0], face[i], face[i+1]],
                            'tex_coords': [tex_face[0], tex_face[i], tex_face[i+1]],
                            'normals': [norm_face[0], norm_face[i], norm_face[i+1]]
                        })

    def load_texture(self, filename):
        self.texture = pygame.image.load(filename)
        self.texture_width = self.texture.get_width()
        self.texture_height = self.texture.get_height()

# Matrices
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
        [f/aspect, 0, 0, 0],
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

# Render
class Renderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.shader = DiscoShader()
        self.light_dir = normalize([1, -1, -1])
        self.zbuffer = np.full((height, width), float('inf'))

    def clear(self):
        self.screen.fill(self.shader.get_sky_color())
        self.zbuffer.fill(float('inf'))

    def render_model(self, model, transform_matrix):
        self.clear()
        for face in model.faces:
            v = [np.array(model.vertices[i]) for i in face['vertices']]
            vt = [model.tex_coords[i] for i in face['tex_coords']]
            vn = [np.array(model.normals[i]) for i in face['normals']]
            v_t = [transform_matrix @ vi for vi in v]
            v_t = [vi / vi[3] if vi[3] != 0 else vi for vi in v_t]
            x, y, z = [vi[0] for vi in v_t], [vi[1] for vi in v_t], [vi[2] for vi in v_t]
            min_x = max(0, int(min(x)))
            max_x = min(self.width - 1, int(max(x)))
            min_y = max(0, int(min(y)))
            max_y = min(self.height - 1, int(max(y)))
            for px in range(min_x, max_x+1):
                for py in range(min_y, max_y+1):
                    try:
                        den = ((y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2]))
                        if abs(den) < 1e-8: continue
                        w1 = ((y[1]-y[2])*(px-x[2])+(x[2]-x[1])*(py-y[2]))/den
                        w2 = ((y[2]-y[0])*(px-x[2])+(x[0]-x[2])*(py-y[2]))/den
                        w3 = 1 - w1 - w2
                        if w1 >= 0 and w2 >= 0 and w3 >= 0:
                            pz = w1*z[0]+w2*z[1]+w3*z[2]
                            if pz < self.zbuffer[py][px]:
                                self.zbuffer[py][px] = pz
                                normal = normalize(w1*vn[0]+w2*vn[1]+w3*vn[2])
                                intensity, depth = self.shader.vertex_shader([px, py, pz], normal, self.light_dir)
                                tx = w1*vt[0][0]+w2*vt[1][0]+w3*vt[2][0]
                                ty = w1*vt[0][1]+w2*vt[1][1]+w3*vt[2][1]
                                tx = int(tx * model.texture_width) % model.texture_width
                                ty = int(ty * model.texture_height) % model.texture_height
                                tex_color = model.texture.get_at((tx, ty))
                                color = self.shader.fragment_shader(intensity, depth, tex_color, px, py)
                                self.screen.set_at((px, py), color)
                    except:
                        continue
        pygame.display.flip()

# Main
def main():
    renderer = Renderer(WIDTH, HEIGHT)
    model = Obj("objetos/casa_madera.obj", "texturas/casa_madera.png")

    eye = [0, 1, 5]
    target = [0, 0, 0]
    up = [0, 1, 0]

    projection = perspective_projection(45, WIDTH/HEIGHT, 0.1, 100)
    view = look_at(eye, target, up)
    viewport = viewport_matrix(0, 0, WIDTH, HEIGHT)
    model_matrix = scale(0.5, 0.5, 0.5) @ translate(0, -0.5, 0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        renderer.shader.time += 0.03
        transform_matrix = viewport @ projection @ view @ model_matrix
        renderer.render_model(model, transform_matrix)
        pygame.time.wait(10)

    pygame.quit()

if __name__ == "__main__":
    main()
