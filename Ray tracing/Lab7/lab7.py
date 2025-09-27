import math, cv2, numpy as np, pygame
from dataclasses import dataclass

WINDOW_W, WINDOW_H = 900, 600
RENDER_W, RENDER_H = 1000, 1000
FOV_DEG = 65
MAX_DEPTH = 5
EPSILON = 1e-3

# ------------------ utilidades ------------------
def normalize(v): n=np.linalg.norm(v); return v if n==0 else v/n
def reflect(I,N): return I-2*np.dot(I,N)*N
def refract(I,N,eta):
    cosi=np.clip(-np.dot(I,N),-1,1); sint2=eta**2*(1-cosi**2)
    if sint2>1: return None
    cost=math.sqrt(max(0,1-sint2)); return eta*I+(eta*cosi-cost)*N
def fresnel_schlick(cos_theta,F0): return F0+(1-F0)*((1-cos_theta)**5)
def to_srgb(img): return np.clip(np.power(np.clip(img,0,1),1/2.2),0,1)

# ------------------ materiales ------------------
@dataclass
class Material:
    diffuse: np.ndarray
    ka: float=0.02; kd: float=0.85; ks: float=0.2; shininess:int=32
    reflection: float=0.0; transparency: float=0.0; ior: float=1.5

# ------------------ primitivas ------------------
@dataclass
class Plane:
    point: np.ndarray; normal: np.ndarray; material: Material
    def intersect(self,ro,rd):
        n=normalize(self.normal); denom=np.dot(n,rd)
        if abs(denom)<1e-6: return None,None
        t=np.dot(self.point-ro,n)/denom
        if t>EPSILON: return t,(n if denom<0 else -n)
        return None,None

@dataclass
class Disk:
    center: np.ndarray; normal: np.ndarray; radius: float; material: Material
    def intersect(self,ro,rd):
        n=normalize(self.normal); denom=np.dot(n,rd)
        if abs(denom)<1e-6: return None,None
        t=np.dot(self.center-ro,n)/denom
        if t<=EPSILON: return None,None
        p=ro+t*rd
        if np.linalg.norm(p-self.center)<=self.radius+1e-6:
            return t,(n if denom<0 else -n)
        return None,None

@dataclass
class Triangle:
    a: np.ndarray; b: np.ndarray; c: np.ndarray; material: Material
    def intersect(self,ro,rd):
        e1=self.b-self.a; e2=self.c-self.a
        pvec=np.cross(rd,e2); det=np.dot(e1,pvec)
        if abs(det)<1e-8: return None,None
        invDet=1.0/det; tvec=ro-self.a; u=np.dot(tvec,pvec)*invDet
        if u<0 or u>1: return None,None
        qvec=np.cross(tvec,e1); v=np.dot(rd,qvec)*invDet
        if v<0 or u+v>1: return None,None
        t=np.dot(e2,qvec)*invDet
        if t<=EPSILON: return None,None
        n=normalize(np.cross(e1,e2)); return t,(n if det<0 else -n)

@dataclass
class Box:
    min_corner: np.ndarray; max_corner: np.ndarray; material: Material
    def intersect(self,ro,rd):
        inv=1.0/np.where(rd!=0,rd,1e-12)
        t0s=(self.min_corner-ro)*inv; t1s=(self.max_corner-ro)*inv
        tmin=np.maximum.reduce(np.minimum(t0s,t1s))
        tmax=np.minimum.reduce(np.maximum(t0s,t1s))
        if tmax<max(tmin,EPSILON): return None,None
        t=tmin if tmin>EPSILON else tmax; p=ro+t*rd
        n=np.array([0,0,0.0])
        for i in range(3):
            if abs(p[i]-self.min_corner[i])<1e-4: n[i]=-1
            elif abs(p[i]-self.max_corner[i])<1e-4: n[i]=1
        n=normalize(n) if np.linalg.norm(n)>0 else np.array([0,0,1.0])
        return t,n

# ------------------ luces ------------------
@dataclass
class PointLight:
    position: np.ndarray; color: np.ndarray; intensity: float; k: float=0.3
    def L_I_and_dist(self,p):
        L=self.position-p; d=np.linalg.norm(L); d=d if d>1e-6 else 1e-6
        L=L/d; atten=1.0/(1.0+self.k*d*d)
        return L,self.color*(self.intensity*atten),d

# ------------------ renderer ------------------
class Renderer:
    def __init__(self,w,h,fov_deg):
        self.w,self.h=w,h; self.aspect=w/h; self.fov=math.radians(fov_deg)
        self.cam=np.array([0,0,0]); self.objects=[]; self.lights=[]
    def add_object(self,o): self.objects.append(o)
    def add_light(self,l): self.lights.append(l)
    def closest_hit(self,ro,rd):
        tmin,hit,nrm=float("inf"),None,None
        for o in self.objects:
            t,n=o.intersect(ro,rd)
            if t and t<tmin: tmin,hit,nrm=t,o,n
        return tmin,hit,nrm
    def shade(self,p,n,v,m):
        col=m.ka*m.diffuse
        for l in self.lights:
            L,I,dist=l.L_I_and_dist(p)
            tS,oS,_=self.closest_hit(p+n*EPSILON,L)
            if oS is not None and tS<dist-EPSILON: continue
            ndotl=max(0.0,np.dot(n,L))
            diff=m.kd*ndotl*(I*m.diffuse)
            R=normalize(2*ndotl*n-L)
            spec=m.ks*(max(0.0,np.dot(R,v))**m.shininess)*I
            col+=diff+spec
        return np.clip(col,0,1.2)
    def trace(self,ro,rd,depth=0):
        if depth>MAX_DEPTH: return np.array([0,0,0])
        t,o,n=self.closest_hit(ro,rd)
        if o is None: return np.array([0.05,0.05,0.07])
        p=ro+t*rd; v=normalize(-rd); m=o.material
        res=self.shade(p,n,v,m)
        if m.reflection>0:
            rdir=normalize(reflect(rd,n))
            rcol=self.trace(p+n*EPSILON,rdir,depth+1)
            res=(1-m.reflection)*res+m.reflection*rcol
        if m.transparency>0:
            nl,cosi=n.copy(),np.dot(rd,n)
            n1,n2=1.0,m.ior
            if cosi>0: nl=-n; n1,n2=n2,n1
            eta=n1/n2; T=refract(rd,nl,eta)
            if T is not None:
                tr=self.trace(p-nl*EPSILON,normalize(T),depth+1)
                F0=((n1-n2)/(n1+n2))**2
                F=fresnel_schlick(abs(np.dot(v,nl)),F0)
                res=(1-m.transparency)*res+(1-F)*m.transparency*tr
        return np.clip(res,0,1.2)
    def render_progressive(self,screen):
        img=np.zeros((self.h,self.w,3),dtype=np.float32)
        scale=math.tan(self.fov/2)
        for y in range(self.h):
            py=(1-2*((y+0.5)/self.h))*scale
            for x in range(self.w):
                px=(2*((x+0.5)/self.w)-1)*scale*self.aspect
                rd=normalize(np.array([px,py,-1.0]))
                img[y,x]=self.trace(self.cam,rd,0)
            preview=(to_srgb(np.clip(img,0,1))*255).astype(np.uint8)
            surf=pygame.surfarray.make_surface(np.swapaxes(preview,0,1))
            surf=pygame.transform.smoothscale(surf,(WINDOW_W,WINDOW_H))
            screen.blit(surf,(0,0)); pygame.display.flip()
        return (to_srgb(np.clip(img,0,1))*255).astype(np.uint8)

# ------------------ escena ------------------
def build_room_scene(r):
    white=Material(np.array([0.92,0.92,0.92]),kd=0.82,ks=0.12,shininess=24,reflection=0.04,ka=0.02)
    gray=Material(np.array([0.6,0.6,0.6]),kd=0.80,ks=0.10,shininess=24,reflection=0.03,ka=0.02)
    red=Material(np.array([0.9,0.25,0.25]),kd=0.88,ks=0.10,shininess=18,ka=0.02)
    green=Material(np.array([0.25,0.82,0.35]),kd=0.88,ks=0.10,shininess=18,ka=0.02)

    gold_reflect=Material(np.array([1.0,0.85,0.3]),kd=0.2,ks=0.9,shininess=256,reflection=0.6,ka=0.01)
    blue_matte=Material(np.array([0.36,0.52,1.0]),kd=0.95,ks=0.06,shininess=8,reflection=0.02,ka=0.02)
    gold_metal=Material(np.array([1.0,0.86,0.35]),kd=0.35,ks=0.85,shininess=128,reflection=0.25,ka=0.01)
    glass=Material(np.array([0.92,0.97,1.0]),kd=0.05,ks=0.5,shininess=96,reflection=0.05,transparency=0.9,ior=1.5,ka=0.01)

    back_z=-9.0; left_x=-4.0; right_x=4.0; floor_y=-2.2; ceil_y=2.8
    r.add_object(Plane(np.array([0,0,back_z]),np.array([0,0,1]),white))
    r.add_object(Plane(np.array([0,floor_y,0]),np.array([0,1,0]),gray))
    r.add_object(Plane(np.array([0,ceil_y,0]),np.array([0,-1,0]),gray))
    r.add_object(Plane(np.array([left_x,0,0]),np.array([1,0,0]),red))
    r.add_object(Plane(np.array([right_x,0,0]),np.array([-1,0,0]),green))

    r.add_object(Box(np.array([-2.4,floor_y,-7.2]),np.array([-1.0,-0.4,-6.0]),gold_reflect))
    r.add_object(Box(np.array([0.8,floor_y,-7.6]),np.array([2.2,0.2,-6.2]),blue_matte))

    tri_a=np.array([-1.0,1.5,back_z+EPSILON]); tri_b=np.array([1.0,2.2,back_z+EPSILON]); tri_c=np.array([0.0,0.6,back_z+EPSILON])
    r.add_object(Triangle(tri_a,tri_b,tri_c,gold_metal))

    disk_center=np.array([-3.3,0.5,-7.0]); disk_normal=np.array([1.0,0.0,0.0])
    r.add_object(Disk(disk_center,disk_normal,0.9,glass))

    # una sola luz
    r.add_light(PointLight(np.array([0,2.6,-6.5]),np.array([1.0,1.0,1.0]),9.0,k=0.32))

def main():
    pygame.init(); screen=pygame.display.set_mode((WINDOW_W,WINDOW_H))
    pygame.display.set_caption("Ray Tracer -Lab 7")
    r=Renderer(RENDER_W,RENDER_H,FOV_DEG); build_room_scene(r)
    img=r.render_progressive(screen)
    import imageio.v2 as imageio; imageio.imwrite("render_lab7.png",img)
    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
    pygame.quit()

if __name__=="__main__": main()
