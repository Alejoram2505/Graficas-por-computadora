import math, numpy as np, pygame
from dataclasses import dataclass
import imageio.v2 as imageio

WINDOW_W, WINDOW_H = 900, 600
RENDER_W, RENDER_H = 300, 300
FOV_DEG = 65
MAX_DEPTH = 5
EPSILON = 1e-3

# ------------------ utilidades ------------------
def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def reflect(I, N):
    return I - 2 * np.dot(I, N) * N

def refract(I, N, eta):
    cosi = np.clip(-np.dot(I, N), -1, 1)
    sint2 = eta ** 2 * (1 - cosi ** 2)
    if sint2 > 1: return None
    cost = math.sqrt(max(0, 1 - sint2))
    return eta * I + (eta * cosi - cost) * N

def fresnel_schlick(cos_theta, F0):
    return F0 + (1 - F0) * ((1 - cos_theta) ** 5)

def to_srgb(img):
    return np.clip(np.power(np.clip(img, 0, 1), 1/2.2), 0, 1)

# ------------------ materiales ------------------
@dataclass
class Material:
    diffuse: np.ndarray
    ka: float = 0.02
    kd: float = 0.85
    ks: float = 0.2
    shininess: int = 32
    reflection: float = 0.0
    transparency: float = 0.0
    ior: float = 1.5

# ------------------ primitivas necesarias ------------------
@dataclass
class Plane:
    point: np.ndarray; normal: np.ndarray; material: Material
    def intersect(self, ro, rd):
        n = normalize(self.normal); denom = np.dot(n, rd)
        if abs(denom) < 1e-6: return None, None
        t = np.dot(self.point - ro, n) / denom
        if t > EPSILON: return t, (n if denom < 0 else -n)
        return None, None

# --------- FIGURA: CÁPSULA ---------
@dataclass
class Capsule:
    a: np.ndarray  # extremo A
    b: np.ndarray  # extremo B
    radius: float
    material: Material
    def intersect(self, ro, rd):
        # Eje de la cápsula
        ba = self.b - self.a
        L = np.linalg.norm(ba)
        if L < 1e-8:
            # degen: tratar como esfera
            oc = ro - self.a
            b = np.dot(oc, rd)
            c = np.dot(oc, oc) - self.radius * self.radius
            disc = b*b - c
            if disc < 0: return None, None
            t = -b - math.sqrt(disc)
            if t <= EPSILON: t = -b + math.sqrt(disc)
            if t <= EPSILON: return None, None
            p = ro + t*rd
            n = normalize(p - self.a)
            return t, n
        baN = ba / L
        # Componentes perpendiculares a baN
        oc = ro - self.a
        rd_par = np.dot(rd, baN)
        oc_par = np.dot(oc, baN)
        rd_perp = rd - rd_par * baN
        oc_perp = oc - oc_par * baN
        A = np.dot(rd_perp, rd_perp)
        B = 2.0 * np.dot(oc_perp, rd_perp)
        C = np.dot(oc_perp, oc_perp) - self.radius * self.radius
        t_cyl = None
        if A > 1e-12:
            disc = B*B - 4*A*C
            if disc >= 0:
                sqrtD = math.sqrt(disc)
                t1 = (-B - sqrtD) / (2*A)
                t2 = (-B + sqrtD) / (2*A)
                for tt in [t1, t2]:
                    if tt > EPSILON:
                        # comprobar si el punto cae dentro del segmento (proyección s)
                        s = oc_par + tt * rd_par
                        if 0.0 <= s <= L:
                            t_cyl = tt if (t_cyl is None or tt < t_cyl) else t_cyl
        # Intersección con las esferas de los extremos
        def ray_sphere(center):
            oc2 = ro - center
            b2 = np.dot(oc2, rd)
            c2 = np.dot(oc2, oc2) - self.radius*self.radius
            disc2 = b2*b2 - c2
            if disc2 < 0: return None
            sD = math.sqrt(disc2)
            tA = -b2 - sD
            tB = -b2 + sD
            return min([t for t in [tA, tB] if t > EPSILON], default=None)
        t_sa = ray_sphere(self.a)
        t_sb = ray_sphere(self.b)
        # Elegir el t más cercano válido
        candidates = [t for t in [t_cyl, t_sa, t_sb] if t is not None]
        if not candidates: return None, None
        t = min(candidates)
        p = ro + t*rd
        # normal
        s = np.dot(p - self.a, baN)
        if 0.0 < s < L and t == t_cyl:
            axis_pt = self.a + s * baN
            n = normalize(p - axis_pt)
        else:
            # esfera más cercana
            if np.linalg.norm(p - self.a) < np.linalg.norm(p - self.b):
                n = normalize(p - self.a)
            else:
                n = normalize(p - self.b)
        return t, n

# ------------------ luces ------------------
@dataclass
class PointLight:
    position: np.ndarray; color: np.ndarray; intensity: float; k: float = 0.3
    def L_I_and_dist(self, p):
        L = self.position - p; d = np.linalg.norm(L); d = d if d > 1e-6 else 1e-6
        L = L / d; atten = 1.0 / (1.0 + self.k * d * d)
        return L, self.color * (self.intensity * atten), d

# ------------------ renderer ------------------
class Renderer:
    def __init__(self, w, h, fov_deg):
        self.w, self.h = w, h; self.aspect = w / h; self.fov = math.radians(fov_deg)
        self.cam = np.array([0, 0, 0]); self.objects = []; self.lights = []
    def add_object(self, o): self.objects.append(o)
    def add_light(self, l): self.lights.append(l)
    def closest_hit(self, ro, rd):
        tmin, hit, nrm = float("inf"), None, None
        for o in self.objects:
            t, n = o.intersect(ro, rd)
            if t and t < tmin: tmin, hit, nrm = t, o, n
        return tmin, hit, nrm
    def shade(self, p, n, v, m):
        col = m.ka * m.diffuse
        for l in self.lights:
            L, I, dist = l.L_I_and_dist(p)
            tS, oS, _ = self.closest_hit(p + n * EPSILON, L)
            if oS is not None and tS < dist - EPSILON: continue
            ndotl = max(0.0, np.dot(n, L))
            diff = m.kd * ndotl * (I * m.diffuse)
            R = normalize(2 * ndotl * n - L)
            spec = m.ks * (max(0.0, np.dot(R, v)) ** m.shininess) * I
            col += diff + spec
        return np.clip(col, 0, 1.2)
    def trace(self, ro, rd, depth=0):
        if depth > MAX_DEPTH: return np.array([0, 0, 0])
        t, o, n = self.closest_hit(ro, rd)
        if o is None: return np.array([0.05, 0.05, 0.07])
        p = ro + t * rd; v = normalize(-rd); m = o.material
        res = self.shade(p, n, v, m)
        if m.reflection > 0:
            rdir = normalize(reflect(rd, n))
            rcol = self.trace(p + n * EPSILON, rdir, depth + 1)
            res = (1 - m.reflection) * res + m.reflection * rcol
        if m.transparency > 0:
            nl, cosi = n.copy(), np.dot(rd, n)
            n1, n2 = 1.0, m.ior
            if cosi > 0: nl = -n; n1, n2 = n2, n1
            eta = n1 / n2; T = refract(rd, nl, eta)
            if T is not None:
                tr = self.trace(p - nl * EPSILON, normalize(T), depth + 1)
                F0 = ((n1 - n2) / (n1 + n2)) ** 2
                F = fresnel_schlick(abs(np.dot(v, nl)), F0)
                res = (1 - m.transparency) * res + (1 - F) * m.transparency * tr
        return np.clip(res, 0, 1.2)
    def render_progressive(self, screen):
        img = np.zeros((self.h, self.w, 3), dtype=np.float32)
        scale = math.tan(self.fov / 2)
        for y in range(self.h):
            py = (1 - 2 * ((y + 0.5) / self.h)) * scale
            for x in range(self.w):
                px = (2 * ((x + 0.5) / self.w) - 1) * scale * self.aspect
                rd = normalize(np.array([px, py, -1.0]))
                img[y, x] = self.trace(self.cam, rd, 0)
            preview = (to_srgb(np.clip(img, 0, 1)) * 255).astype(np.uint8)
            surf = pygame.surfarray.make_surface(np.swapaxes(preview, 0, 1))
            surf = pygame.transform.smoothscale(surf, (WINDOW_W, WINDOW_H))
            screen.blit(surf, (0, 0)); pygame.display.flip()
        return (to_srgb(np.clip(img, 0, 1)) * 255).astype(np.uint8)

# ------------------ escena ------------------

def build_room_scene(r):
    # materiales base
    white = Material(np.array([0.92, 0.92, 0.92]), kd=0.82, ks=0.12, shininess=24, reflection=0.04, ka=0.02)
    gray  = Material(np.array([0.60, 0.60, 0.60]), kd=0.80, ks=0.10, shininess=24, reflection=0.03, ka=0.02)
    red   = Material(np.array([0.90, 0.25, 0.25]), kd=0.88, ks=0.10, shininess=18, ka=0.02)
    green = Material(np.array([0.25, 0.82, 0.35]), kd=0.88, ks=0.10, shininess=18, ka=0.02)

    # cuarto
    back_z = -9.0; left_x = -4.0; right_x = 4.0; floor_y = -2.2; ceil_y = 2.8
    r.add_object(Plane(np.array([0, 0, back_z]), np.array([0, 0, 1]), white))
    r.add_object(Plane(np.array([0, floor_y, 0]), np.array([0, 1, 0]), gray))
    r.add_object(Plane(np.array([0, ceil_y, 0]), np.array([0, -1, 0]), gray))
    r.add_object(Plane(np.array([left_x, 0, 0]), np.array([1, 0, 0]), red))
    r.add_object(Plane(np.array([right_x, 0, 0]), np.array([-1, 0, 0]), green))

    # -------- CÁPSULAS (3 variantes) --------
    matte    = Material(np.array([0.92, 0.74, 0.62]), kd=0.90, ks=0.22, shininess=64, reflection=0.03, ka=0.02)
    mirrorish= Material(np.array([0.55, 0.75, 0.95]), kd=0.22, ks=0.85, shininess=220, reflection=0.65, ka=0.01)
    glassy   = Material(np.array([0.85, 0.98, 0.92]), kd=0.05, ks=0.55, shininess=130, reflection=0.05, transparency=0.92, ior=1.45, ka=0.005)

    r.add_object(Capsule(np.array([-0.80, -1.40, -6.60]), np.array([-1.00,  0.80, -6.90]), 0.36, matte))          # opaca
    r.add_object(Capsule(np.array([ 0.2, -1.6, -7.2]), np.array([ 1.3,  0.0, -6.8]), 0.30, mirrorish))            # reflectiva
    r.add_object(Capsule(np.array([ 2.1, -1.1, -7.6]), np.array([ 2.7,  1.0, -7.9]), 0.26, glassy))               # transparente

    # luz (una puntual cenital con menos hotspot)
    r.add_light(PointLight(np.array([0, 2.6, -6.5]), np.array([1.0, 1.0, 1.0]), 6.0, k=0.45))


def main():
    pygame.init(); screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Ray Tracer - Lab 8")
    r = Renderer(RENDER_W, RENDER_H, FOV_DEG); build_room_scene(r)
    img = r.render_progressive(screen)
    imageio.imwrite("render_lab8_capsula.png", img)
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
    pygame.quit()

if __name__ == "__main__":
    main()
