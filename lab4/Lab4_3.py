import pygame
import numpy as np
import math

WIDTH, HEIGHT = 800, 800
BLACK = (0, 0, 0)

# Utilidades
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def look_at(eye, target, up):
    forward = normalize(np.array(target) - np.array(eye))
    right = normalize(np.cross(forward, up))
    new_up = normalize(np.cross(right, forward))
    return np.array([
        [*right, -np.dot(right, eye)],
        [*new_up, -np.dot(new_up, eye)],
        [*-forward, np.dot(forward, eye)],
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

# Cargador de modelo OBJ
class Obj:
    def __init__(self, filename):
        self.vertices = []
        self.normals = []
        self.faces = []
        self.load_model(filename)

    def load_model(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    _, x, y, z = line.split()
                    self.vertices.append([float(x), float(y), float(z), 1])
                elif line.startswith('vn '):
                    _, nx, ny, nz = line.split()
                    self.normals.append([float(nx), float(ny), float(nz)])
                elif line.startswith('f '):
                    face = []
                    normal_face = []
                    parts = line.strip().split()[1:]
                    for part in parts:
                        v_idx, _, n_idx = (part.split('/') + ['0'])[:3]
                        face.append(int(v_idx) - 1)
                        if n_idx:
                            normal_face.append(int(n_idx) - 1)
                    if len(face) >= 3:
                        for i in range(1, len(face)-1):
                            self.faces.append({
                                'vertices': [face[0], face[i], face[i+1]],
                                'normals': [normal_face[0], normal_face[i], normal_face[i+1]] if normal_face else None
                            })

# Shader reflectivo tipo diamante
class DiamondShader:
    def __init__(self):
        self.base_color = np.array([230, 230, 255])  # Azul claro casi blanco
        self.shininess = 512  # Brillo muy alto para reflejo intenso

    def vertex_shader(self, position, normal, light_dir, view_dir):
        normal = normalize(normal)
        light_dir = normalize(light_dir)
        view_dir = normalize(view_dir)

        diffuse = max(np.dot(normal, light_dir), 0.0)
        reflect_dir = normalize(2 * np.dot(normal, light_dir) * normal - light_dir)
        specular = np.power(max(np.dot(view_dir, reflect_dir), 0.0), self.shininess)

        # Reflejo tipo espejo con tonalidad azul metálica
        final_color = (
            self.base_color * (0.3 + 0.7 * diffuse) +
            np.array([255, 255, 255]) * specular * 1.2
        )

        return np.clip(final_color, 0, 255)

# Renderer
class Renderer:
    def __init__(self, width, height):
        pygame.init()
        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height))
        self.zbuffer = np.full((height, width), float('inf'))
        self.shader = DiamondShader()
        self.light_dir = np.array([1, -1, -1])
        self.view_pos = np.array([0, 1, 5])

    def clear(self):
        self.screen.fill(BLACK)
        self.zbuffer.fill(float('inf'))

    def render_model(self, model, transform):
        self.clear()
        for face in model.faces:
            v = [np.array(model.vertices[i]) for i in face['vertices']]
            n = [np.array(model.normals[i]) if face['normals'] else np.array([0, 0, 1]) for i in range(3)]

            vt = [transform @ vert for vert in v]
            for i in range(3):
                if vt[i][3] != 0:
                    vt[i] = vt[i] / vt[i][3]

            x, y, z = [vtx[:3] for vtx in vt]
            min_x, max_x = max(0, int(min(x[0], y[0], z[0]))), min(self.width-1, int(max(x[0], y[0], z[0])))
            min_y, max_y = max(0, int(min(x[1], y[1], z[1]))), min(self.height-1, int(max(x[1], y[1], z[1])))

            for i in range(min_x, max_x+1):
                for j in range(min_y, max_y+1):
                    denom = ((y[1]-z[1])*(x[0]-z[0]) + (z[0]-y[0])*(x[1]-z[1]))
                    if abs(denom) < 1e-6: continue
                    w1 = ((y[1]-z[1])*(i-z[0]) + (z[0]-y[0])*(j-z[1])) / denom
                    w2 = ((z[1]-x[1])*(i-z[0]) + (x[0]-z[0])*(j-z[1])) / denom
                    w3 = 1 - w1 - w2
                    if min(w1, w2, w3) < 0: continue
                    z_interp = w1*x[2] + w2*y[2] + w3*z[2]
                    if z_interp < self.zbuffer[j][i]:
                        self.zbuffer[j][i] = z_interp
                        interp_normal = normalize(w1*n[0] + w2*n[1] + w3*n[2])
                        frag_pos = w1*x + w2*y + w3*z
                        view_dir = normalize(self.view_pos - frag_pos)
                        color = self.shader.vertex_shader(frag_pos, interp_normal, self.light_dir, view_dir)
                        self.screen.set_at((i, j), tuple(color.astype(int)))
        pygame.display.flip()

# Main
def main():
    renderer = Renderer(WIDTH, HEIGHT)
    model = Obj("objetos/casa_madera.obj")

    eye = [0, 1, 5]
    target = [0, 0, 0]
    up = [0, 1, 0]

    projection = perspective_projection(45, WIDTH / HEIGHT, 0.1, 100)
    view = look_at(eye, target, up)
    viewport = viewport_matrix(0, 0, WIDTH, HEIGHT)

    model_matrix = scale(0.5, 0.5, 0.5) @ translate(0, -0.5, 0)
    transform = viewport @ projection @ view @ model_matrix

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        renderer.render_model(model, transform)
        pygame.time.wait(10)

    pygame.quit()

if __name__ == "__main__":
    main()
