import pygame
import numpy as np
from PIL import Image
import math

# Configuración de la ventana
WIDTH = 800
HEIGHT = 800
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Obj:
    def __init__(self, filename, texture_filename):
        self.vertices = []
        self.tex_coords = []
        self.faces = []
        self.load_model(filename)
        self.load_texture(texture_filename)
        
    def load_model(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):  # vertices
                    _, x, y, z = line.split()
                    self.vertices.append([float(x), float(y), float(z), 1])
                elif line.startswith('vt '):  # texture coordinates
                    _, u, v = line.split()
                    self.tex_coords.append([float(u), 1 - float(v)])
                elif line.startswith('f '):  # faces
                    face = []
                    tex_face = []
                    parts = line.strip().split()[1:]
                    for part in parts:
                        v_idx, t_idx, *_ = (part.split('/') + ['0'])[:3]
                        face.append(int(v_idx) - 1)
                        if t_idx:
                            tex_face.append(int(t_idx) - 1)
                    if len(face) >= 3:
                        # Triangulación de caras
                        for i in range(1, len(face)-1):
                            self.faces.append({
                                'vertices': [face[0], face[i], face[i+1]],
                                'tex_coords': [tex_face[0], tex_face[i], tex_face[i+1]] if tex_face else None
                            })
    
    def load_texture(self, filename):
        self.texture = pygame.image.load(filename)
        self.texture_width = self.texture.get_width()
        self.texture_height = self.texture.get_height()

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def look_at(eye, target, up):
    forward = normalize(np.array(target) - np.array(eye))
    right = normalize(np.cross(forward, up))
    new_up = normalize(np.cross(right, forward))
    
    view_matrix = np.array([
        [right[0], right[1], right[2], -np.dot(right, eye)],
        [new_up[0], new_up[1], new_up[2], -np.dot(new_up, eye)],
        [-forward[0], -forward[1], -forward[2], np.dot(forward, eye)],
        [0, 0, 0, 1]
    ])
    return view_matrix

def perspective_projection(fov, aspect, near, far):
    f = 1 / math.tan(math.radians(fov) / 2)
    projection_matrix = np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ])
    return projection_matrix

def viewport_matrix(x, y, width, height):
    return np.array([
        [width/2, 0, 0, x + width/2],
        [0, -height/2, 0, y + height/2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def rotate_x(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

def rotate_y(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])

def rotate_z(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
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

class Renderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.zbuffer = np.full((height, width), float('inf'))
        
    def clear(self):
        self.screen.fill(BLACK)
        self.zbuffer.fill(float('inf'))
        
    def point(self, x, y, z, color):
        if 0 <= x < self.width and 0 <= y < self.height:
            if z < self.zbuffer[int(y)][int(x)]:
                self.zbuffer[int(y)][int(x)] = z
                self.screen.set_at((int(x), int(y)), color)

    def render_model(self, model, model_matrix, view_matrix, projection_matrix):
        self.clear()
        
        transform_matrix = viewport_matrix(0, 0, self.width, self.height) @ projection_matrix @ view_matrix @ model_matrix
        
        for face in model.faces:
            v1 = np.array(model.vertices[face['vertices'][0]])
            v2 = np.array(model.vertices[face['vertices'][1]])
            v3 = np.array(model.vertices[face['vertices'][2]])
            
            # Transformar vértices
            v1_transformed = transform_matrix @ v1
            v2_transformed = transform_matrix @ v2
            v3_transformed = transform_matrix @ v3
            
            # Perspectiva division
            if v1_transformed[3] != 0:
                v1_transformed = v1_transformed / v1_transformed[3]
            if v2_transformed[3] != 0:
                v2_transformed = v2_transformed / v2_transformed[3]
            if v3_transformed[3] != 0:
                v3_transformed = v3_transformed / v3_transformed[3]
            
            x1, y1, z1 = v1_transformed[:3]
            x2, y2, z2 = v2_transformed[:3]
            x3, y3, z3 = v3_transformed[:3]
            
            # Calcular bounding box
            min_x = max(0, min(x1, x2, x3))
            max_x = min(self.width - 1, max(x1, x2, x3))
            min_y = max(0, min(y1, y2, y3))
            max_y = min(self.height - 1, max(y1, y2, y3))
            
            # Rasterización y texturizado
            for x in range(int(min_x), int(max_x) + 1):
                for y in range(int(min_y), int(max_y) + 1):
                    # Coordenadas baricéntricas
                    w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / \
                         ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
                    w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / \
                         ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
                    w3 = 1 - w1 - w2
                    
                    if w1 >= 0 and w2 >= 0 and w3 >= 0:
                        # Interpolación de Z
                        z = w1 * z1 + w2 * z2 + w3 * z3
                        
                        if z < self.zbuffer[y][x]:
                            self.zbuffer[y][x] = z
                            
                            if face['tex_coords']:
                                # Interpolación de coordenadas de textura
                                tx1, ty1 = model.tex_coords[face['tex_coords'][0]]
                                tx2, ty2 = model.tex_coords[face['tex_coords'][1]]
                                tx3, ty3 = model.tex_coords[face['tex_coords'][2]]
                                
                                tx = w1 * tx1 + w2 * tx2 + w3 * tx3
                                ty = w1 * ty1 + w2 * ty2 + w3 * ty3
                                
                                # Obtener color de textura
                                tx = int(tx * model.texture_width) % model.texture_width
                                ty = int(ty * model.texture_height) % model.texture_height
                                color = model.texture.get_at((tx, ty))
                                
                                self.screen.set_at((x, y), color)
                            else:
                                self.screen.set_at((x, y), WHITE)
        
        pygame.display.flip()

    def save_screenshot(self, filename):
        pygame.image.save(self.screen, filename)

def main():
    renderer = Renderer(WIDTH, HEIGHT)
    model = Obj("objetos/casa_madera.obj", "texturas/casa_madera.png")
    
    # Configuración de cámara
    fov = 60
    aspect = WIDTH / HEIGHT
    near = 0.1
    far = 100
    
    projection = perspective_projection(fov, aspect, near, far)
    
    # Configuraciones de cámara para diferentes tomas
    camera_configs = {
        "medium_shot": {
            "eye": [0, 0, 6],
            "target": [0, 0, 0],
            "up": [0, 1, 0],
            "model_rotation": [0, 0, 0],
            "scale": 0.8
        },
        "low_angle": {
            "eye": [0, -2, 6],
            "target": [0, 1, 0],
            "up": [0, 1, 0],
            "model_rotation": [0, 0, 0],
            "scale": 0.8
        },
        "high_angle": {
            "eye": [0, 2, 6],
            "target": [0, -1, 0],
            "up": [0, 1, 0],
            "model_rotation": [0, 0, 0],
            "scale": 0.8
        },
        "dutch_angle": {
            "eye": [0, 0, 6],
            "target": [0, 0, 0],
            "up": [math.sin(math.radians(30)), math.cos(math.radians(30)), 0],
            "model_rotation": [0, math.radians(15), 0],
            "scale": 0.8
        }
    }
    
    # Tomar las cuatro fotos
    for shot_name, config in camera_configs.items():
        # Matriz de vista
        view = look_at(
            config["eye"],
            config["target"],
            config["up"]
        )
        
        # Matriz de modelo
        model_matrix = np.eye(4)
        model_matrix = model_matrix @ rotate_x(config["model_rotation"][0])
        model_matrix = model_matrix @ rotate_y(config["model_rotation"][1])
        model_matrix = model_matrix @ rotate_z(config["model_rotation"][2])
        model_matrix = model_matrix @ scale(config.get("scale", 1), config.get("scale", 1), config.get("scale", 1))
        model_matrix = model_matrix @ translate(0, -1, 0)  # Ajustar posición vertical
        
        # Renderizar y guardar
        renderer.render_model(model, model_matrix, view, projection)
        renderer.save_screenshot(f"{shot_name}.png")
        pygame.time.wait(1000)  # Esperar 1 segundo entre tomas
    
    pygame.quit()

if __name__ == "__main__":
    main()
