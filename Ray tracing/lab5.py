import pygame
import numpy as np
from dataclasses import dataclass, field

# =========================================================
# AJUSTA AQUÍ EL TAMAÑO DE LA TORTUGA
# =========================================================
SCENE_SCALE = 0.60   # escala global 
SCENE_Z     = -4.2   # hacia atrás

# FOV de cámara: más grande = abre la vista
FOV_DEG = 50

# Resoluciones
WINDOW_W, WINDOW_H = 700, 700       # ventana pygame
RENDER_W, RENDER_H = 520, 520       # resolución del ray tracer (se escala a la ventana)

# =========================================================
# Utilidades
# =========================================================
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

# =========================================================
# Material (Phong)
# =========================================================
@dataclass
class Material:
    diffuse: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.2, 0.2]))
    ka: float = 0.12  # ambiente
    kd: float = 0.85  # difuso
    ks: float = 0.5   # especular
    shininess: int = 48

# =========================================================
# Geometría: Esfera
# =========================================================
@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    material: Material

    # Intersección rayo-esfera -> (t, normal) o (None, None)
    def intersect(self, ray_origin: np.ndarray, ray_dir: np.ndarray):
        oc = ray_origin - self.center
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - self.radius * self.radius
        disc = b*b - 4.0*c
        if disc < 0.0:
            return None, None
        sqrt_disc = np.sqrt(disc)
        t0 = (-b - sqrt_disc) / 2.0
        t1 = (-b + sqrt_disc) / 2.0
        candidates = [t for t in (t0, t1) if t > 1e-4]
        if not candidates:
            return None, None
        t = min(candidates)
        p = ray_origin + t * ray_dir
        n = normalize(p - self.center)
        return t, n

# =========================================================
# Luz direccional
# =========================================================
@dataclass
class DirectionalLight:
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    intensity: float = 1.0
    # vector que apunta DESDE la luz hacia la escena
    direction: np.ndarray = field(default_factory=lambda: np.array([-0.5, -1.0, -0.8]))

    def get(self):
        # Devuelve (L, I): dirección unit hacia la luz desde el punto e intensidad
        return normalize(-self.direction), self.color * self.intensity

# =========================================================
# Renderizador
# =========================================================
class Renderer:
    def __init__(self, width=512, height=512, fov_deg=50):
        self.width = width
        self.height = height
        self.aspect = width / height
        self.fov = np.deg2rad(fov_deg)
        self.camera_pos = np.array([0.0, 0.0, 0.0])
        self.objects = []
        self.lights = []
        self.bgcolor = np.array([0.95, 0.96, 0.92])  

    def add_object(self, obj): self.objects.append(obj)
    def add_light(self, light): self.lights.append(light)

    def trace(self, ro, rd):
        hit_t = float("inf")
        hit_obj = None
        hit_n = None
        for obj in self.objects:
            t, n = obj.intersect(ro, rd)
            if t is not None and t < hit_t:
                hit_t, hit_obj, hit_n = t, obj, n
        return hit_t, hit_obj, hit_n

    def shade(self, p, n, v, material):
        # Phong: ka*Ia + sum( kd*(L·N)*Il + ks*(R·V)^s * Il )
        Ia = np.ones(3)  
        color = material.ka * Ia

        for light in self.lights:
            L, I = light.get()

            # Sombras: rayo hacia la luz direccional
            shadow_t, shadow_obj, _ = self.trace(p + n * 1e-3, L)
            if shadow_obj is not None:
                continue

            ndotl = max(0.0, float(np.dot(n, L)))
            diffuse = material.kd * ndotl * (I * material.diffuse)

            R = normalize(2.0 * ndotl * n - L)
            rdotv = max(0.0, float(np.dot(R, v)))
            specular = material.ks * (rdotv ** material.shininess) * I

            color += diffuse + specular

        return np.clip(color, 0.0, 1.0)

    def render(self):
        W, H = self.width, self.height
        img = np.zeros((H, W, 3), dtype=np.float32)
        scale = np.tan(self.fov * 0.5)

        for y in range(H):
            py = (1.0 - 2.0 * ((y + 0.5) / H)) * scale
            for x in range(W):
                px = (2.0 * ((x + 0.5) / W) - 1.0) * scale * self.aspect
                rd = normalize(np.array([px, py, -1.0]))  
                ro = self.camera_pos

                t, obj, n = self.trace(ro, rd)
                if obj is None:
                    img[y, x] = self.bgcolor
                else:
                    p = ro + t * rd
                    v = normalize(-rd)
                    img[y, x] = self.shade(p, n, v, obj.material)

        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

# =========================================================
# Escena: Tortuga hecha con esferas (usa escala y Z)
# =========================================================
def build_turtle_scene(renderer, scale=1.0, z=-3.2):
    # Materiales
    skin = Material(diffuse=np.array([0.58, 0.78, 0.24]), ka=0.18, kd=0.85, ks=0.35, shininess=32)
    shell = Material(diffuse=np.array([0.88, 0.73, 0.55]), ka=0.15, kd=0.85, ks=0.55, shininess=64)

    # Geometría (todo se multiplica por 'scale')
    R_shell = 1.05 * scale
    renderer.add_object(Sphere(center=np.array([0.0, 0.0, z]), radius=R_shell, material=shell))

    # Cabeza
    R_head = 0.32 * scale
    renderer.add_object(Sphere(center=np.array([0.0, R_shell + R_head * 0.85, z]),
                               radius=R_head, material=skin))

    # Patas
    R_leg = 0.24 * scale
    off = R_shell * 0.82
    for lx, ly in [(-off, -off), (off, -off), (-off, off), (off, off)]:
        renderer.add_object(Sphere(center=np.array([lx, ly, z]), radius=R_leg, material=skin))

    # Cola
    R_tail = 0.18 * scale
    renderer.add_object(Sphere(center=np.array([0.0, -R_shell - R_tail * 0.9, z]),
                               radius=R_tail, material=skin))

    # Luz direccional (tipo sol)
    renderer.add_light(DirectionalLight(
        color=np.array([1.0, 1.0, 1.0]),
        intensity=1.0,
        direction=np.array([-0.7, -1.1, -0.9])
    ))

# =========================================================
# Visualización 
# =========================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Ray Tracer - Tortuga (solo esferas)")
    clock = pygame.time.Clock()

    renderer = Renderer(width=RENDER_W, height=RENDER_H, fov_deg=FOV_DEG)
    build_turtle_scene(renderer, scale=SCENE_SCALE, z=SCENE_Z)  

    img = renderer.render()  
    surface = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))
    surface = pygame.transform.smoothscale(surface, (WINDOW_W, WINDOW_H))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
