import pygame
import os
import numpy as np

# CONFIGURACIÓN
WIDTH, HEIGHT = 1000, 700
FPS = 30
OBJ_DIR = "objetos"
TEX_DIR = "texturas"

# CÁMARA Y PROYECCIÓN
def perspective_projection(fov, aspect, near, far):
    f = 1 / np.tan(np.radians(fov) / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])

def look_at(eye, target, up):
    f = (target - eye)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    view = np.eye(4)
    view[0, :3] = r
    view[1, :3] = u
    view[2, :3] = -f
    view[:3, 3] = -view[:3, :3] @ eye
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

# OBJ LOADER
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
                        self.texcoords.append((self._f(parts[1]), 1 - self._f(parts[2])))
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

# FUNCIONES DE RENDER
def barycentric_coords(p, a, b, c):
    denom = ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
    if denom == 0:
        return -1, -1, -1
    w1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
    w2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
    w3 = 1 - w1 - w2
    return w1, w2, w3

def draw_obj(screen, zbuffer, obj, texture, solid_color, model_matrix, view, proj):
    tex_array = pygame.surfarray.pixels3d(texture) if texture else None
    for i, tri in enumerate(obj.faces_v):
        pts = []
        z_values = []
        for vi in tri:
            x, y, z = transform_vertex(obj.vertices[vi], model_matrix, view, proj)
            pts.append((x, y))
            z_values.append(z)

        if len(pts) != 3:
            continue

        if texture and obj.faces_vt[i]:
            tex_coords = [obj.texcoords[ti] for ti in obj.faces_vt[i]]
            min_x = max(min(p[0] for p in pts), 0)
            max_x = min(max(p[0] for p in pts), WIDTH - 1)
            min_y = max(min(p[1] for p in pts), 0)
            max_y = min(max(p[1] for p in pts), HEIGHT - 1)

            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    w1, w2, w3 = barycentric_coords((x, y), *pts)
                    if min(w1, w2, w3) >= 0:
                        z = w1 * z_values[0] + w2 * z_values[1] + w3 * z_values[2]
                        if z < zbuffer[x][y]:
                            zbuffer[x][y] = z
                            u = w1 * tex_coords[0][0] + w2 * tex_coords[1][0] + w3 * tex_coords[2][0]
                            v = w1 * tex_coords[0][1] + w2 * tex_coords[1][1] + w3 * tex_coords[2][1]
                            tx = int(u * texture.get_width())
                            ty = int(v * texture.get_height())
                            if 0 <= tx < texture.get_width() and 0 <= ty < texture.get_height():
                                color = tex_array[tx, ty]
                                screen.set_at((x, y), color)
        elif solid_color:
            pygame.draw.polygon(screen, solid_color, pts)
        else:
            pygame.draw.polygon(screen, (180, 180, 180), pts)

# MAIN LOOP
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Proyecto1 - Escena Texturas")
    clock = pygame.time.Clock()

    fondo = pygame.image.load("fondo.png").convert()
    fondo = pygame.transform.scale(fondo, (WIDTH, HEIGHT))

    casa = OBJ(os.path.join(OBJ_DIR, "casa_nieve.obj"))
    casita = OBJ(os.path.join(OBJ_DIR, "casita.obj"))
    banca = OBJ(os.path.join(OBJ_DIR, "banca.obj"))
    planta = OBJ(os.path.join(OBJ_DIR, "Planta.obj"))

    tex_casa = pygame.image.load(os.path.join(TEX_DIR, "casa_nieve.jpg")).convert()
    tex_casita = pygame.image.load(os.path.join(TEX_DIR, "casita.png")).convert()
    tex_planta = pygame.image.load(os.path.join(TEX_DIR, "planta.jpg")).convert()

    eye = np.array([0, 0, 10], dtype=np.float32)
    target = np.array([0, 0, 0], dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    view = look_at(eye, target, up)
    proj = perspective_projection(45, WIDTH / HEIGHT, 0.1, 100)

    running = True
    while running:
        zbuffer = np.full((WIDTH, HEIGHT), float('inf'))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(fondo, (0, 0))

        draw_obj(screen, zbuffer, casa, tex_casa, None, np.eye(4) @ scale_matrix(0.03) @ rotate_y_matrix(200) @ translate_matrix(120, -70, 1), view, proj)
        draw_obj(screen, zbuffer, casita, tex_casita, None, np.eye(4) @ scale_matrix(0.1) @ translate_matrix(10, -2.6, 1), view, proj)
        draw_obj(screen, zbuffer, banca, None, (150, 100, 50), np.eye(4) @ scale_matrix(0.015) @ translate_matrix(190, -220, 5), view, proj)
        draw_obj(screen, zbuffer, planta, tex_planta, None, np.eye(4) @ scale_matrix(0.6) @ translate_matrix(1.5, -5, 3), view, proj)

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
