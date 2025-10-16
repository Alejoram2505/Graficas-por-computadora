import math, numpy as np, pygame, imageio.v2 as imageio
from dataclasses import dataclass

# ---------------- CONFIG ----------------
WINDOW_W, WINDOW_H = 800, 500
RENDER_W, RENDER_H = 700, 700
FOV_DEG = 60
MAX_DEPTH = 5
EPSILON = 1e-3
ENVIRONMENT_MAP_PATH = "metro.hdr"

# ---------------- UTILS -----------------
def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def reflect(I, N): return I - 2*np.dot(I, N)*N

def refract(I, N, eta):
    cosi = np.clip(-np.dot(I, N), -1, 1)
    sint2 = eta**2 * (1 - cosi**2)
    if sint2 > 1: return None
    cost = math.sqrt(1 - sint2)
    return eta*I + (eta*cosi - cost)*N

def fresnel_schlick(cos_theta, F0): return F0 + (1 - F0)*((1 - cos_theta)**5)
def to_srgb(img): return np.clip(np.power(np.clip(img, 0, 1), 1/2.2), 0, 1)

def make_basis(n):
    n = normalize(n)
    a = np.array([0, 1, 0]) if abs(n[2]) > 0.999 else np.array([0, 0, 1])
    t = normalize(np.cross(a, n)); b = np.cross(n, t)
    return t, b, n

# ---------------- MATERIAL -----------------
@dataclass
class Material:
    diffuse: np.ndarray
    ka: float = 0.02; kd: float = 0.8; ks: float = 0.3; shininess: int = 96
    reflection: float = 0.0; transparency: float = 0.0; ior: float = 1.5
    emisive: float = 0.0; tex: np.ndarray | None = None

# ---------------- FIGURES ------------------
@dataclass
class Plane:
    point: np.ndarray; normal: np.ndarray; material: Material
    def intersect(self, ro, rd):
        n = normalize(self.normal)
        denom = np.dot(n, rd)
        if abs(denom) < 1e-8: return None, None
        t = np.dot(self.point - ro, n) / denom
        if t > EPSILON: return t, (n if denom < 0 else -n)
        return None, None
    def uv(self, p):
        return (p[0]*0.3)%1.0, (p[2]*0.3)%1.0

@dataclass
class Sphere:
    center: np.ndarray; radius: float; material: Material
    def intersect(self, ro, rd):
        oc=ro-self.center; b=2*np.dot(oc,rd); c=np.dot(oc,oc)-self.radius**2
        disc=b*b-4*c
        if disc<0:return None,None
        s=math.sqrt(disc); t0,t1=(-b-s)/2,(-b+s)/2
        t=t0 if t0>EPSILON else (t1 if t1>EPSILON else None)
        if t is None:return None,None
        p=ro+t*rd; n=normalize(p-self.center)
        return t,n

@dataclass
class Cylinder:
    center: np.ndarray; radius: float; half_height: float; material: Material
    def intersect(self, ro, rd):
        oc=ro-self.center
        A=rd[0]**2+rd[2]**2; B=2*(oc[0]*rd[0]+oc[2]*rd[2]); C=oc[0]**2+oc[2]**2-self.radius**2
        disc=B*B-4*A*C
        if disc<0:return None,None
        s=math.sqrt(disc)
        for tt in [(-B-s)/(2*A),(-B+s)/(2*A)]:
            if tt>EPSILON:
                y=oc[1]+tt*rd[1]
                if -self.half_height<=y<=self.half_height:
                    p=ro+tt*rd
                    n=normalize(np.array([p[0]-self.center[0],0,p[2]-self.center[2]]))
                    return tt,n
        return None,None

@dataclass
class Cone:
    apex: np.ndarray; height: float; radius: float; material: Material
    def intersect(self, ro, rd):
        k=self.radius/self.height; k2=k*k; oc=ro-self.apex
        A=rd[0]**2+rd[2]**2-k2*rd[1]**2
        B=2*(oc[0]*rd[0]+oc[2]*rd[2]-k2*oc[1]*rd[1])
        C=oc[0]**2+oc[2]**2-k2*oc[1]**2
        disc=B*B-4*A*C
        if disc<0:return None,None
        s=math.sqrt(disc)
        for tt in [(-B-s)/(2*A),(-B+s)/(2*A)]:
            if tt>EPSILON:
                y=oc[1]+tt*rd[1]
                if -self.height<=y<=0:
                    p=ro+tt*rd
                    dx,dz=p[0]-self.apex[0],p[2]-self.apex[2]
                    n=normalize(np.array([dx,k*math.sqrt(dx*dx+dz*dz),dz]))
                    return tt,n
        return None,None

@dataclass
class Ellipsoid:
    center: np.ndarray; axes: np.ndarray; material: Material
    def intersect(self, ro, rd):
        oc=ro-self.center; inv2=1.0/(self.axes*self.axes)
        A=np.dot(rd*rd,inv2); B=2*np.dot(oc*rd,inv2); C=np.dot(oc*oc,inv2)-1
        disc=B*B-4*A*C
        if disc<0:return None,None
        s=math.sqrt(disc); t0,t1=(-B-s)/(2*A),(-B+s)/(2*A)
        t=t0 if t0>EPSILON else (t1 if t1>EPSILON else None)
        if t is None:return None,None
        p=ro+t*rd; n=normalize((p-self.center)/(self.axes*self.axes))
        return t,n

@dataclass
class Torus:
    center: np.ndarray; normal: np.ndarray; R: float; r: float; material: Material
    def intersect(self, ro, rd):
        t,b,n=make_basis(self.normal)
        o=ro-self.center
        O=np.array([np.dot(o,t),np.dot(o,b),np.dot(o,n)])
        D=np.array([np.dot(rd,t),np.dot(rd,b),np.dot(rd,n)])
        px2=np.array([O[0]**2,2*O[0]*D[0],D[0]**2])
        py2=np.array([O[1]**2,2*O[1]*D[1],D[1]**2])
        pz2=np.array([O[2]**2,2*O[2]*D[2],D[2]**2])
        s=px2+py2+pz2
        g=s.copy(); g[0]+=self.R**2-self.r**2
        g2=np.polynomial.polynomial.polymul(g,g)
        h=px2+py2; h_pad=np.zeros(5); h_pad[:3]=h
        f=g2-(4*self.R*self.R)*h_pad
        roots=np.roots(f[::-1])
        ts=[float(r.real) for r in roots if abs(r.imag)<1e-6 and r.real>EPSILON]
        if not ts:return None,None
        t_hit=min(ts)
        P=O+t_hit*D
        g0=(P[0]**2+P[1]**2+P[2]**2)+self.R**2-self.r**2
        grad=4*g0*P-8*(self.R**2)*np.array([P[0],P[1],0])
        n_local=normalize(grad)
        n_world=normalize(n_local[0]*t+n_local[1]*b+n_local[2]*n)
        return t_hit,n_world

@dataclass
class EnergyDisk:
    center: np.ndarray; normal: np.ndarray; radius: float; material: Material
    def intersect(self, ro, rd):
        n=normalize(self.normal); denom=np.dot(n,rd)
        if abs(denom)<1e-8:return None,None
        t=np.dot(self.center-ro,n)/denom
        if t<EPSILON:return None,None
        p=ro+t*rd
        if np.linalg.norm(p-self.center)>self.radius:return None,None
        return t,n

# ---------------- LIGHTS ------------------
@dataclass
class SpotLight:
    position: np.ndarray
    direction: np.ndarray
    color: np.ndarray
    intensity: float
    inner_deg: float
    outer_deg: float
    k: float = 0.30
    def sample(self, p):
        Lvec = self.position - p
        d = np.linalg.norm(Lvec); d = d if d>1e-6 else 1e-6
        L = Lvec / d
        spot_dir = normalize(self.direction)
        cos_theta = np.dot(-L, spot_dir)
        inner = math.cos(math.radians(self.inner_deg))
        outer = math.cos(math.radians(self.outer_deg))
        if cos_theta <= outer: return L, np.array([0,0,0]), d
        t = np.clip((cos_theta - outer) / (inner - outer + 1e-8), 0, 1)
        atten = 1.0 / (1.0 + self.k * d * d)
        return L, self.color * (self.intensity * t * atten), d

@dataclass
class PointLight:
    position: np.ndarray; color: np.ndarray; intensity: float; k: float=0.35
    def sample(self, p):
        L=self.position-p; d=np.linalg.norm(L); L=L/(d+1e-6)
        atten=1.0/(1.0+self.k*d*d)
        return L,self.color*(self.intensity*atten),d

@dataclass
class DirectionalLight:
    direction: np.ndarray; color: np.ndarray; intensity: float
    def sample(self, p):
        L=normalize(-self.direction); return L,self.color*self.intensity,float("inf")

# ---------------- RENDERER -----------------
class Renderer:
    def __init__(self,w,h,fov_deg):
        self.w,self.h=w,h; self.aspect=w/h; self.fov=math.radians(fov_deg)
        self.cam=np.array([0,1.2,-12.0])
        self.objects,self.lights=[],[]
        self.load_env(ENVIRONMENT_MAP_PATH)
    def load_env(self,path):
        try:
            arr=imageio.imread(path)
            if arr.dtype!=np.float32: arr=arr.astype(np.float32)/255.0
            self.env=arr; print("[OK] Env:", arr.shape)
        except Exception as e:
            print("ENV ERROR",e); self.env=None
    def env_color(self,rd):
        if self.env is None:return np.array([0.05,0.06,0.08])
        d=normalize(rd)
        phi=math.atan2(d[2],d[0]); u=(phi+math.pi)/(2*math.pi)
        v=0.5-(math.asin(np.clip(d[1],-1,1))/math.pi)
        h,w=self.env.shape[:2]; x,y=int(u*(w-1)),int(v*(h-1))
        return np.clip(self.env[y,x],0,1)
    def closest_hit(self,ro,rd):
        tmin=float("inf"); obj=None; nrm=None
        for o in self.objects:
            t,n=o.intersect(ro,rd)
            if t and t<tmin: tmin,obj,nrm=t,o,n
        return tmin,obj,nrm
    def shade(self,p,n,v,obj):
        m=obj.material; base=m.diffuse
        if isinstance(obj,Plane) and m.tex is not None:
            u,vtex=obj.uv(p); H,W=m.tex.shape[:2]
            x=int(u*(W-1)); y=int((1-vtex)*(H-1)); base=m.tex[y,x]
        col=m.ka*base
        for Lsrc in self.lights:
            L,I,dist=Lsrc.sample(p)
            tS,oS,_=self.closest_hit(p+n*EPSILON,L)
            if oS and tS<dist-EPSILON: continue
            ndotl=max(0,np.dot(n,L))
            diff=m.kd*ndotl*(I*base)
            R=normalize(reflect(-L,n))
            spec=m.ks*((max(0,np.dot(R,v)))**m.shininess)*I
            col+=diff+spec
        if m.emisive>0: col+=base*m.emisive
        return np.clip(col,0,1.5)
    def trace(self,ro,rd,depth=0):
        if depth>MAX_DEPTH:return self.env_color(rd)
        t,o,n=self.closest_hit(ro,rd)
        if o is None:return self.env_color(rd)
        p=ro+t*rd; v=normalize(-rd)
        col=self.shade(p,n,v,o)
        m=o.material
        if m.reflection>0:
            rdir=normalize(reflect(rd,n))
            rcol=self.trace(p+n*EPSILON,rdir,depth+1)
            col=(1-m.reflection)*col+m.reflection*rcol
        if m.transparency>0:
            n1,n2=1.0,m.ior; nl=n if np.dot(rd,n)<0 else -n
            eta=n1/n2 if np.dot(rd,n)<0 else n2/n1
            T=refract(rd,nl,eta)
            if T is not None:
                tr=self.trace(p-nl*EPSILON,normalize(T),depth+1)
                F0=((n1-n2)/(n1+n2))**2; F=fresnel_schlick(abs(np.dot(v,nl)),F0)
                col=(1-m.transparency)*col+(1-F)*m.transparency*tr
        return np.clip(col,0,1.2)
    def render_progressive(self,screen):
        img=np.zeros((self.h,self.w,3),dtype=np.float32)
        scale=math.tan(self.fov/2)
        for y in range(self.h):
            py=(1-2*((y+0.5)/self.h))*scale
            for x in range(self.w):
                px=(2*((x+0.5)/self.w)-1)*scale*self.aspect
                rd=normalize(np.array([px,py,1.0]))
                img[y,x]=self.trace(self.cam,rd)
            surf=pygame.surfarray.make_surface((to_srgb(img)*255).astype(np.uint8).swapaxes(0,1))
            surf=pygame.transform.smoothscale(surf,(WINDOW_W,WINDOW_H))
            screen.blit(surf,(0,0)); pygame.display.flip()
        return (to_srgb(np.clip(img,0,1))*255).astype(np.uint8)

# ---------------- SCENE -----------------
def build_portal_scene(r: Renderer):
    # Materials
    metal_dark=Material(np.array([0.3,0.32,0.36]),ks=0.9,shininess=220,reflection=0.55)
    metal_light=Material(np.array([0.86,0.9,0.95]),ks=0.95,shininess=256,reflection=0.48)
    glass_blue=Material(np.array([0.55,0.8,1.0]),kd=0.12,ks=0.9,shininess=200,reflection=0.12,transparency=0.9,ior=1.5)
    energy=Material(np.array([0.18,0.72,1.0]),emisive=0.7,ka=0.35,kd=0.9)
    lamp_white=Material(np.array([1.0,1.0,0.9]),emisive=0.8)

    # Floor
    H,W=256,256; tex=np.zeros((H,W,3),dtype=np.float32)
    for j in range(H):
        for i in range(W):
            c=((i//16)+(j//16))&1
            tex[j,i]=[0.09,0.09,0.10] if c==0 else [0.15,0.15,0.16]
    floor_mat=Material(np.array([0.12,0.12,0.13]),reflection=0.4,ks=0.25,kd=0.85,tex=tex)
    r.objects.append(Plane(np.array([0,-3.9,0]),np.array([0,1,0]),floor_mat))

    # Portal main body
    center=np.array([0,1.1,6.2]); axis=np.array([0,0,1]); R_big=4.6
    r.objects.append(Torus(center,axis,R_big,0.5,metal_light))
    r.objects.append(Torus(center,axis,3.0,0.35,glass_blue))
    r.objects.append(Torus(center,axis,2.1,0.28,energy))
    r.objects.append(EnergyDisk(center,axis,2.0,energy))

    # cylinders
    r.objects.append(Cylinder(np.array([-5.3,0,6.2]),0.50,5.3,metal_dark))
    r.objects.append(Cylinder(np.array([ 5.3,0,6.2]),0.50,5.3,metal_dark))

    # Cones config
    lamp_apex_y = 10.8        
    lamp_spacing = 7.2        
    lamp_height = 1.8         
    lamp_radius = 1.2         

    # Material del bulbo emisivo
    lamp_bulb = Material(np.array([1.0, 0.95, 0.85]), emisive=0.9, kd=0.2, ks=0.0)

    for x in [-lamp_spacing, 0.0, lamp_spacing]:
        # Cono metálico (la lámpara)
        r.objects.append(Cone(
            np.array([x, lamp_apex_y, 6.0]),
            height=lamp_height,
            radius=lamp_radius,
            material=metal_dark
        ))

        # Bulbo pequeño dentro del cono
        bulb_y = lamp_apex_y - (lamp_height - 0.50)
        r.objects.append(Sphere(np.array([x, bulb_y, 6.0]), 0.30, lamp_bulb))

        # Luz cálida tipo foco (SpotLight)
        r.lights.append(SpotLight(
            position=np.array([x, bulb_y, 6.0]),
            direction=np.array([0, -1, 0]),       
            color=np.array([1.0, 0.92, 0.78]),
            intensity= 80.0,                   
            inner_deg=8, outer_deg=15,
            k=0.35
        ))

    # Ellipsoides 
    for ang in np.linspace(0, 2*math.pi, 12, endpoint=False):
        ex = center[0] + math.cos(ang) * R_big * 1.03
        ey = center[1] + math.sin(ang) * R_big * 1.03
        ez = center[2]
        r.objects.append(Ellipsoid(np.array([ex,ey,ez]), np.array([0.55,0.25,0.55]), lamp_white))

    # Ondas de energía (esferas)
    waves = [
        (-2.2, 0.6, 5.4, 0.35), (-1.0, 1.5, 5.2, 0.28),
        ( 0.0, 0.9, 5.0, 0.32), ( 1.2, 1.6, 5.3, 0.27),
        ( 2.0, 0.7, 5.5, 0.30), (-0.6, 2.0, 5.1, 0.22)
    ]
    for x,y,z,radius in waves:
        r.objects.append(Sphere(np.array([x,y,z]), radius, glass_blue))

    # Lights nucleo y ambiente
    r.lights.append(PointLight(center, np.array([0.65,0.85,1.0]), 58.0, k=0.40))
    r.lights.append(PointLight(center+np.array([0,3.0,1.2]), np.array([0.70,0.90,1.0]), 18.0, k=0.60))
    r.lights.append(PointLight(center+np.array([-4.0, 1.0, -1.0]), np.array([0.60,0.80,1.0]), 16.0, k=0.55))
    r.lights.append(PointLight(center+np.array([ 4.0, 1.0, -1.0]), np.array([0.60,0.80,1.0]), 16.0, k=0.55))
    r.lights.append(DirectionalLight(np.array([-0.28,-1.0,-0.40]), np.array([1.0,0.90,0.80]), 0.35))

# ---------------- MAIN -------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Ray Tracer – Portal Sci-Fi ")
    r = Renderer(RENDER_W, RENDER_H, FOV_DEG)
    build_portal_scene(r)
    img = r.render_progressive(screen)
    imageio.imwrite("render_proy2.png", img)
    print("Guardado: render_proy2.png")
    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
    pygame.quit()

if __name__ == "__main__":
    main()
