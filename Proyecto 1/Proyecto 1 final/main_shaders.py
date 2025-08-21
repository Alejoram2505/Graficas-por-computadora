import pygame
import os
import numpy as np

from casa_madera_shader import CRTGlitchShader
from casita_shader import NeonShader
from banca_shader import StaticNoiseShader
from planta_shader import GradientShader

WIDTH, HEIGHT = 1000, 700
FPS = 30
OBJ_DIR = "objetos"
TEX_DIR = "texturas"

def perspective_projection(fov, aspect, near, far):
    f = 1 / np.tan(np.radians(fov) / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])

def look_at(eye, target, up):
    f = np.array(target, dtype=np.float64) - np.array(eye, dtype=np.float64)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    view = np.eye(4)
    view[0, :3] = r
    view[1, :3] = u
    view[2, :3] = -f
    view[:3, 3] = -np.dot(view[:3, :3], eye)
    return view

def transform_vertex(v, model, view, proj):
    vertex = np.array([*v, 1])
    transformed = proj @ view @ model @ vertex
    if transformed[3] != 0:
        transformed /= transformed[3]
    x = int((transformed[0] + 1) * WIDTH / 2)
    y = int((1 - transformed[1]) * HEIGHT / 2)
    z = transformed[2]
    return (x, y, z)

class OBJ:
    def __init__(self, path):
        self.vertices = []
        self.texcoords = []
        self.faces_v = []
        self.faces_vt = []
        self._load(path)

    def _f(self, x):
        try: return float(x)
        except: return 0.0

    def _load(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    _, x, y, z = (line + " 0 0").split()[:4]
                    self.vertices.append((self._f(x), self._f(y), self._f(z)))
                elif line.startswith("vt "):
                    parts = line.split()
                    if len(parts) >= 3:
                        self.texcoords.append((self._f(parts[1]), self._f(parts[2])))
                elif line.startswith("f "):
                    toks = line.split()[1:]
                    v_idx, t_idx = [], []
                    for t in toks:
                        parts = t.split('/')
                        v = int(parts[0]) - 1
                        v_idx.append(v)
                        if len(parts) > 1 and parts[1] != '':
                            t = int(parts[1]) - 1
                            t_idx.append(t)
                    for i in range(1, len(v_idx) - 1):
                        self.faces_v.append([v_idx[0], v_idx[i], v_idx[i+1]])
                        if len(t_idx) >= 3:
                            self.faces_vt.append([t_idx[0], t_idx[i], t_idx[i+1]])
                        else:
                            self.faces_vt.append(None)

def draw_obj(screen, z_buffer, obj, texture, shader, model_matrix, view, proj):
    for i, tri in enumerate(obj.faces_v):
        pts = []
        for vi in tri:
            v = obj.vertices[vi]
            pts.append(transform_vertex(v, model_matrix, view, proj))

        if len(pts) != 3:
            continue

        avg_z = sum(p[2] for p in pts) / 3
        x_avg = sum(p[0] for p in pts) // 3
        y_avg = sum(p[1] for p in pts) // 3

        if 0 <= x_avg < WIDTH and 0 <= y_avg < HEIGHT:
            if avg_z < z_buffer[y_avg][x_avg]:
                z_buffer[y_avg][x_avg] = avg_z

                if shader:
                    if isinstance(shader, StaticNoiseShader):
                        color = shader.vertex_shader(pts[0], pts[0][0], pts[0][1])
                        pygame.draw.polygon(screen, color, [(p[0], p[1]) for p in pts])

                    elif isinstance(shader, GradientShader):
                        min_y = min(p[1] for p in pts)
                        max_y = max(p[1] for p in pts)
                        height = max_y - min_y
                        center_y = (min_y + max_y) // 2
                        color = shader.vertex_shader((0, center_y), pts[0][0], center_y, height)
                        pygame.draw.polygon(screen, color, [(p[0], p[1]) for p in pts])

                    elif isinstance(shader, CRTGlitchShader):
                        shader.draw_wireframe(screen, [(p[0], p[1]) for p in pts])

                    elif isinstance(shader, NeonShader):
                        for y in range(min(p[1] for p in pts), max(p[1] for p in pts)):
                            for x in range(min(p[0] for p in pts), max(p[0] for p in pts)):
                                if screen.get_rect().collidepoint(x, y):
                                    c = shader.vertex_shader(pts[0], [0, 0, 1], [1, -1, -1], [0, 0, 1], x, y)
                                    screen.set_at((x, y), tuple(c))
                else:
                    pygame.draw.polygon(screen, (180, 180, 180), [(p[0], p[1]) for p in pts])

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Proyecto1 - Escena Completa con Shaders")
    clock = pygame.time.Clock()

    fondo = pygame.image.load("fondo.png").convert()
    fondo = pygame.transform.scale(fondo, (WIDTH, HEIGHT))

    casa = OBJ(os.path.join(OBJ_DIR, "casa_nieve.obj"))
    casita = OBJ(os.path.join(OBJ_DIR, "casita.obj"))
    banca = OBJ(os.path.join(OBJ_DIR, "banca.obj"))
    planta = OBJ(os.path.join(OBJ_DIR, "Planta.obj"))

    shader_casa = CRTGlitchShader()
    shader_casita = NeonShader()
    shader_banca = StaticNoiseShader()
    shader_planta = GradientShader()

    eye = np.array([0, 0, 10])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    view = look_at(eye, target, up)
    proj = perspective_projection(45, WIDTH / HEIGHT, 0.1, 100)

    running = True
    while running:
        z_buffer = np.full((HEIGHT, WIDTH), np.inf)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(fondo, (0, 0))

        draw_obj(screen, z_buffer, casa, None, shader_casa, np.eye(4) @ scale_matrix(0.03) @ rotate_y_matrix(200) @ translate_matrix(120, -70, 1), view, proj)
        draw_obj(screen, z_buffer, casita, None, shader_casita, np.eye(4) @ scale_matrix(0.1) @ translate_matrix(10, -2.6, 1), view, proj)
        draw_obj(screen, z_buffer, banca, None, shader_banca, np.eye(4) @ scale_matrix(0.015) @ translate_matrix(190, -220, 5), view, proj)
        draw_obj(screen, z_buffer, planta, None, shader_planta, np.eye(4) @ scale_matrix(0.6) @ translate_matrix(1.5, -5, 3), view, proj)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

def rotate_y_matrix(angle_degrees):
    angle = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([
        [ cos_a, 0, sin_a, 0],
        [     0, 1,     0, 0],
        [-sin_a, 0, cos_a, 0],
        [     0, 0,     0, 1]
    ])

def translate_matrix(x, y, z):
    m = np.eye(4)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m

def scale_matrix(s):
    return np.diag([s, s, s, 1])

if __name__ == "__main__":
    main()
