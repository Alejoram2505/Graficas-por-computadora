import pygame
import os
import numpy as np
from shaders.shader_casa_madera import SimpleLineShader
from shaders.shader_planta import GradientShader
from shaders.shader_casita import CRTGlitchShader
from shaders.shader_banca import StaticNoiseShader
        
class OBJ:
    def __init__(self, path):
        self.vertices = []     
        self.texcoords = []    
        self.faces_v = []      
        self.faces_vt = []     
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


def edge(p0, p1, x, y):
    return (x - p0[0]) * (p1[1] - p0[1]) - (y - p0[1]) * (p1[0] - p0[0])

def rasterize_triangle(surface, pts, uvs, texture):
    
    tex_w, tex_h = texture.get_width(), texture.get_height()
    min_x = max(min(p[0] for p in pts), 0)
    max_x = min(max(p[0] for p in pts), surface.get_width()-1)
    min_y = max(min(p[1] for p in pts), 0)
    max_y = min(max(p[1] for p in pts), surface.get_height()-1)

    p0, p1, p2 = pts
    area = edge(p0, p1, p2[0], p2[1])
    if area == 0:
        return
    bias0 = 0 if area < 0 else -1e-5
    bias1 = 0 if area < 0 else -1e-5
    bias2 = 0 if area < 0 else -1e-5

    tex_pixels = pygame.surfarray.pixels3d(texture)

    for y in range(min_y, max_y+1):
        for x in range(min_x, max_x+1):
            w0 = edge(p1, p2, x, y) + bias0
            w1 = edge(p2, p0, x, y) + bias1
            w2 = edge(p0, p1, x, y) + bias2
            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                w0 /= area
                w1 /= area
                w2 /= area

                u = uvs[0][0]*w0 + uvs[1][0]*w1 + uvs[2][0]*w2
                v = uvs[0][1]*w0 + uvs[1][1]*w1 + uvs[2][1]*w2
                tx = int(max(0, min(tex_w-1, u * tex_w)))
                ty = int(max(0, min(tex_h-1, (1.0 - v) * tex_h)))
                color = tex_pixels[tx, ty]
                surface.set_at((x, y), (int(color[0]), int(color[1]), int(color[2])))

    del tex_pixels

# ============================================================
# Proyección ortográfica y helpers de dibujo
# ============================================================
def project_point(v, pos, scale):
    x, y, z = v
    sx = int(x * scale + pos[0])
    sy = int(-y * scale + pos[1])
    return (sx, sy)

def draw_obj_textured(screen, obj, pos, scale, texture, shader=None):
    """Dibuja el objeto texturizado si hay UVs; si no hay textura usa color sólido."""
    
    if texture is None:
        for tri in obj.faces_v:
            pts = [project_point(obj.vertices[i], pos, scale) for i in tri]
            if len(pts) == 3:
                if shader:
                    if isinstance(shader, SimpleLineShader):
                        pygame.draw.polygon(screen, (101, 67, 33), pts, 0)  
                        min_x = min(p[0] for p in pts)
                        max_x = max(p[0] for p in pts)
                        min_y = min(p[1] for p in pts)
                        max_y = max(p[1] for p in pts)
                        for y in range(min_y, max_y + 1):
                            if y % 4 == 0:
                                for x in range(min_x, max_x + 1):
                                    if screen.get_rect().collidepoint(x, y):
                                        color = shader.vertex_shader((x, y), x, y)
                                        screen.set_at((x, y), color)
                    elif isinstance(shader, GradientShader):
                        min_y = min(p[1] for p in pts)
                        max_y = max(p[1] for p in pts)
                        height = max_y - min_y
                        center_y = (min_y + max_y) // 2
                        color = shader.vertex_shader((0, center_y), pts[0][0], center_y, height)
                        pygame.draw.polygon(screen, color, pts)
                    elif isinstance(shader, StaticNoiseShader):
                        color = shader.vertex_shader(pts[0], pts[0][0], pts[0][1])
                        pygame.draw.polygon(screen, color, pts)
                    else:
                        pygame.draw.polygon(screen, (101, 67, 33), pts)
                else:
                    pygame.draw.polygon(screen, (101, 67, 33), pts)
        return

    has_uv = any(fuv is not None for fuv in obj.faces_vt)
    if not has_uv:
        draw_obj_wire(screen, obj, pos, scale, color=(101, 67, 33), fill=True)
        return

    tex_surf = texture.convert()
    for tri_v, tri_vt in zip(obj.faces_v, obj.faces_vt):
        if tri_vt is None:
            pts = [project_point(obj.vertices[i], pos, scale) for i in tri_v]
            if len(pts) == 3:
                pygame.draw.polygon(screen, (101, 67, 33), pts, 0)  
                pygame.draw.polygon(screen, (76, 50, 25), pts, 1)   
            continue

        pts = [project_point(obj.vertices[i], pos, scale) for i in tri_v]
        uvs = []
        for ti in tri_vt:
            if 0 <= ti < len(obj.texcoords):
                uvs.append(obj.texcoords[ti])
            else:
                uvs.append((0.0,0.0))
        if len(pts) == 3 and len(uvs) == 3:
            if shader and isinstance(shader, CRTGlitchShader):
                min_x = min(p[0] for p in pts)
                max_x = max(p[0] for p in pts)
                min_y = min(p[1] for p in pts)
                max_y = max(p[1] for p in pts)
                for y in range(min_y, max_y + 1):
                    for x in range(min_x, max_x + 1):
                        if screen.get_rect().collidepoint(x, y):
                            color = shader.vertex_shader(pts[0], [0,0,1], [1,-1,-1], [0,0,1], x, y)
                            screen.set_at((x, y), tuple(color))
            else:
                rasterize_triangle(screen, pts, uvs, tex_surf)
            

def draw_obj_wire(screen, obj, pos, scale, color=(0,0,0), fill=False):
    for tri in obj.faces_v:
        pts = [project_point(obj.vertices[i], pos, scale) for i in tri]
        if len(pts) == 3:
            if fill:
                pygame.draw.polygon(screen, color, pts, 0)  
                pygame.draw.polygon(screen, (0,0,0), pts, 1)  
            else:
                pygame.draw.polygon(screen, color, pts, 1)

# ============================================================
# MAIN
# ============================================================
def main():
    pygame.init()
    WIDTH, HEIGHT = 1000, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Proyecto - Software Raster (sin OpenGL)")

    # Fondo
    fondo = pygame.image.load("fondo.png").convert()
    fondo = pygame.transform.scale(fondo, (WIDTH, HEIGHT))

    # Cargar modelos
    OBJ_DIR = "objetos"
    TEX_DIR = "texturas"

    casa_madera = OBJ(os.path.join(OBJ_DIR, "casa_madera.obj"))
    casita      = OBJ(os.path.join(OBJ_DIR, "casita.obj"))
    banca       = OBJ(os.path.join(OBJ_DIR, "banca.obj"))
    planta      = OBJ(os.path.join(OBJ_DIR, "Planta.obj"))

    # Cargar texturas (si no existe alguna, queda None )
    def tex(name):
        path = os.path.join(TEX_DIR, name)
        return pygame.image.load(path).convert() if os.path.exists(path) else None

    # Cargar shaders
    shader_casa_madera = SimpleLineShader()  # Líneas para la casa de madera
    shader_casita = CRTGlitchShader()       # Glitch para la casita
    shader_planta = GradientShader()         # Gradiente para las plantas
    shader_banca = StaticNoiseShader()       # Ruido estático para las bancas
    
    tex_casa_madera = tex("casa_madera.png")
    tex_casita      = tex("casita.png")
    tex_banca       = None  
    tex_planta      = tex("planta.jpg")

    # Tus posiciones y escalas exactas
    objetos = [
        (casa_madera, tex_casa_madera, (690, 400), 15, shader_casa_madera),
        (casa_madera, tex_casa_madera, (430, 320), 18, None),  
        (casita,      tex_casita,      (580, 380),  5, shader_casita),
        (casita,      tex_casita,      (920, 310),  5, None),  

        (banca,       tex_banca,       (550, 580), 1.2, None),
        (banca,       tex_banca,       (800, 570), 1.2, shader_banca),

        (planta,      None,      (420, 595), 60, shader_planta),
        (planta,      tex_planta,      (940, 575), 62, None),
    ]

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(fondo, (0, 0))

        for obj, tex_s, pos, scale, shader in objetos:
            draw_obj_textured(screen, obj, pos, scale, tex_s, shader)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()