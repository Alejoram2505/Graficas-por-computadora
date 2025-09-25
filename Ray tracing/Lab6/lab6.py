
import math, time
import cv2
import numpy as np
import pygame
from dataclasses import dataclass, field

# ==============================
# Configuración
# ==============================
WINDOW_W, WINDOW_H = 700, 700
RENDER_W, RENDER_H = 500, 500
FOV_DEG = 70
MAX_DEPTH = 4
EPSILON = 1e-3

ENVIRONMENT_MAP_PATH = "fondo.hdr"   

FLIP_U, FLIP_V = False, False

# ==============================
# Utilidades
# ==============================
def normalize(v): return v / np.linalg.norm(v) if np.linalg.norm(v) else v
def reflect(I,N): return I - 2*np.dot(I,N)*N
def refract(I,N,eta):
    cosi = np.clip(-np.dot(I,N),-1,1)
    sint2 = eta**2*(1-cosi**2)
    if sint2>1: return None
    cost = math.sqrt(max(0,1-sint2))
    return eta*I + (eta*cosi-cost)*N
def fresnel_schlick(cos_theta,F0): return F0+(1-F0)*((1-cos_theta)**5)

# ==============================
# Material
# ==============================
@dataclass
class Material:
    diffuse: np.ndarray
    ka: float = 0.15
    kd: float = 0.8
    ks: float = 0.5
    shininess: int = 48
    reflection: float = 0.0
    transparency: float = 0.0
    ior: float = 1.5

# ==============================
# Esfera
# ==============================
@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    material: Material
    def intersect(self, ro, rd):
        oc = ro - self.center
        b = 2*np.dot(oc,rd)
        c = np.dot(oc,oc)-self.radius**2
        disc = b*b-4*c
        if disc<0: return None,None
        s=math.sqrt(disc)
        t0,t1=(-b-s)/2,(-b+s)/2
        t=None
        if t0>EPSILON: t=t0
        elif t1>EPSILON: t=t1
        if t is None: return None,None
        p=ro+t*rd
        n=normalize(p-self.center)
        return t,n

# ==============================
# Luz
# ==============================
@dataclass
class DirectionalLight:
    direction: np.ndarray
    color: np.ndarray
    intensity: float
    def L_and_I(self): return normalize(-self.direction), self.color*self.intensity

# ==============================
# Renderer
# ==============================
class Renderer:
    def __init__(self,w,h,fov_deg):
        self.w,self.h=w,h
        self.aspect=w/h
        self.fov=math.radians(fov_deg)
        self.cam=np.array([0,0,0])
        self.objects,self.lights=[],[]
        self.env=None
        self.load_env(ENVIRONMENT_MAP_PATH)

    def add_object(self,o): self.objects.append(o)
    def add_light(self,l): self.lights.append(l)


    def load_env(self, path):
        try:
            if path.endswith(".hdr"):
                arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)  
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)    
                arr = np.clip(arr, 0, None)                   
                arr = arr / (1.0 + arr)
                self.env = arr.astype(np.float32)
            else:
                surf = pygame.image.load(path).convert()
                arr = pygame.surfarray.array3d(surf)
                arr = np.transpose(arr, (1,0,2)) / 255.0
                self.env = arr.astype(np.float32)
            print(f"[OK] Cargado environment map {path} {self.env.shape}")
        except Exception as e:
            print(f"[ERROR] {e}")
            self.env = None



    def env_color(self, rd):
        if self.env is None: return np.array([0.2,0.5,1.0])
        d = normalize(rd)
        phi = math.atan2(d[2], d[0])
        u = (phi + math.pi) / (2*math.pi)
        v = 0.5 - (math.asin(np.clip(d[1],-1,1))/math.pi)
        if FLIP_U: u = 1-u
        if FLIP_V: v = 1-v
        h,w = self.env.shape[:2]
        x = u*(w-1); y = v*(h-1)
        x0,y0 = int(x), int(y)
        x1,y1 = min(x0+1,w-1), min(y0+1,h-1)
        dx,dy = x-x0, y-y0
        c00 = self.env[y0,x0]
        c10 = self.env[y0,x1]
        c01 = self.env[y1,x0]
        c11 = self.env[y1,x1]
        c0 = c00*(1-dx)+c10*dx
        c1 = c01*(1-dx)+c11*dx
        return c0*(1-dy)+c1*dy

    def closest_hit(self,ro,rd):
        tmin,hit,nrm=float("inf"),None,None
        for o in self.objects:
            t,n=o.intersect(ro,rd)
            if t and t<tmin: tmin,hit,nrm=t,o,n
        return tmin,hit,nrm

    def shade(self,p,n,v,m):
        col=m.ka*m.diffuse
        for l in self.lights:
            L,I=l.L_and_I()
            if self.closest_hit(p+n*EPSILON,L)[1] is not None: continue
            ndotl=max(0,np.dot(n,L))
            diff=m.kd*ndotl*(I*m.diffuse)
            R=normalize(2*ndotl*n-L)
            spec=m.ks*(max(0,np.dot(R,v))**m.shininess)*I
            col+=diff+spec
        return np.clip(col,0,1)

    def trace(self,ro,rd,depth=0):
        if depth>MAX_DEPTH: return self.env_color(rd)
        t,o,n=self.closest_hit(ro,rd)
        if o is None: return self.env_color(rd)
        p=ro+t*rd; v=normalize(-rd); m=o.material
        local=self.shade(p,n,v,m); res=local.copy()
        if m.reflection>0:
            rdir=normalize(reflect(rd,n))
            rcol=self.trace(p+n*EPSILON,rdir,depth+1)
            res=(1-m.reflection)*res+m.reflection*rcol
        if m.transparency>0:
            nl,cosi=n.copy(),np.dot(rd,n)
            n1,n2=1.0,m.ior
            if cosi>0: nl=-n; n1,n2=n2,n1
            eta=n1/n2
            T=refract(rd,nl,eta)
            if T is not None:
                tr=self.trace(p-nl*EPSILON,normalize(T),depth+1)
                F0=((n1-n2)/(n1+n2))**2
                F=fresnel_schlick(abs(np.dot(v,nl)),F0)
                res=(1-m.transparency)*res+(1-F)*m.transparency*tr
        return np.clip(res,0,1)

    def render_progressive(self,screen):
        img=np.zeros((self.h,self.w,3),dtype=np.uint8)
        scale=math.tan(self.fov/2)
        for y in range(self.h):
            py=(1-2*((y+0.5)/self.h))*scale
            for x in range(self.w):
                px=(2*((x+0.5)/self.w)-1)*scale*self.aspect
                rd=normalize(np.array([px,py,-1]))
                col=self.trace(self.cam,rd,0)
                img[y,x]=(np.clip(col,0,1)*255).astype(np.uint8)
            surf=pygame.surfarray.make_surface(np.swapaxes(img,0,1))
            surf=pygame.transform.smoothscale(surf,(WINDOW_W,WINDOW_H))
            screen.blit(surf,(0,0)); pygame.display.flip()
        return img

# ==============================
# Escena
# ==============================
def build_scene(r):
    z=-5.2;R=0.75;dy=1.35;dx=1.85
    opaque1=Material(np.array([0.7,0.7,0.2]))
    opaque2=Material(np.array([0.3,0.2,0.4]))
    mirror=Material(np.array([1,1,1]),reflection=0.95)
    gold=Material(np.array([1.0,0.85,0.3]),reflection=0.7)
    glass_blue=Material(np.array([0.6,0.8,1.0]),reflection=0.05,transparency=0.92,ior=1.5)
    glass_clear=Material(np.array([1,1,1]),reflection=0.1,transparency=0.95,ior=1.5)
    r.add_object(Sphere(np.array([-dx, dy,z]),R,opaque1))
    r.add_object(Sphere(np.array([ 0 , dy,z]),R,opaque2))
    r.add_object(Sphere(np.array([ dx, dy,z]),R,mirror))
    r.add_object(Sphere(np.array([-dx,-dy,z]),R,gold))
    r.add_object(Sphere(np.array([ 0 ,-dy,z]),R,glass_blue))
    r.add_object(Sphere(np.array([ dx,-dy,z]),R,glass_clear))
    r.add_light(DirectionalLight(np.array([-0.6,-1,-0.8]),np.array([1,1,1]),1.0))

# ==============================
# Main
# ==============================
def main():
    pygame.init()
    screen=pygame.display.set_mode((WINDOW_W,WINDOW_H))
    pygame.display.set_caption("Ray Tracer - Esferas")
    r=Renderer(RENDER_W,RENDER_H,FOV_DEG)
    build_scene(r)
    r.render_progressive(screen)
    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
    pygame.quit()

if __name__=="__main__":
    main()
