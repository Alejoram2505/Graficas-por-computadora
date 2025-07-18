import pygame
import math
import random

WIDTH, HEIGHT = 800, 800
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def load_obj(path):
    vertices = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.strip().split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]
                if len(face) >= 3:
                    # triangulamos en caso de caras con más de 3 vértices
                    for i in range(1, len(face) - 1):
                        faces.append((face[0], face[i], face[i+1]))
    return vertices, faces

def rotate(vertex, angles):
    x, y, z = vertex
    rx, ry, rz = angles

    cosx, sinx = math.cos(rx), math.sin(rx)
    y, z = y * cosx - z * sinx, y * sinx + z * cosx

    cosy, siny = math.cos(ry), math.sin(ry)
    x, z = x * cosy + z * siny, -x * siny + z * cosy

    cosz, sinz = math.cos(rz), math.sin(rz)
    x, y = x * cosz - y * sinz, x * sinz + y * cosz

    return [x, y, z]

def project(v, distance):
    fov = 256
    factor = fov / (distance + v[2])
    x = int(WIDTH / 2 + v[0] * factor)
    y = int(HEIGHT / 2 - v[1] * factor)
    return x, y, v[2]

def normalize_model(vertices, scale_up=2.5):
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    cz = (max(zs) + min(zs)) / 2
    size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2

    norm = []
    for v in vertices:
        norm.append([
            (v[0] - cx) / size * scale_up,
            (v[1] - cy) / size * scale_up,
            (v[2] - cz) / size * scale_up
        ])
    return norm

def draw_triangle(screen, zbuffer, pts, z_vals, color):
    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]
    min_x = max(min(x_coords), 0)
    max_x = min(max(x_coords), WIDTH - 1)
    min_y = max(min(y_coords), 0)
    max_y = min(max(y_coords), HEIGHT - 1)

    def barycentric(A, B, C, P):
        det = ((B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1]))
        if det == 0:
            return -1, -1, -1
        u = ((B[1] - C[1]) * (P[0] - C[0]) + (C[0] - B[0]) * (P[1] - C[1])) / det
        v = ((C[1] - A[1]) * (P[0] - C[0]) + (A[0] - C[0]) * (P[1] - C[1])) / det
        w = 1 - u - v
        return u, v, w

    for x in range(int(min_x), int(max_x) + 1):
        for y in range(int(min_y), int(max_y) + 1):
            u, v, w = barycentric(pts[0], pts[1], pts[2], (x, y))
            if u >= 0 and v >= 0 and w >= 0:
                z = min(z_vals)
                if zbuffer[y][x] is None or z < zbuffer[y][x]:
                    zbuffer[y][x] = z
                    screen.set_at((x, y), color)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("OBJ Build Animation - Vagabond")

    vertices, faces = load_obj("Vagabond.obj")
    vertices = normalize_model(vertices, scale_up=2.0)

    triangle_colors = [
        (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        for _ in faces
    ]

    angle = [0, 0, 0]
    zoom = 3.5
    clock = pygame.time.Clock()
    running = True
    start_time = pygame.time.get_ticks()

    while running:
        dt = pygame.time.get_ticks() - start_time
        mode = 'points' if dt < 3000 else 'wire' if dt < 6000 else 'solid'

        clock.tick(30)
        screen.fill(BLACK)
        zbuffer = [[None for _ in range(WIDTH)] for _ in range(HEIGHT)]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: angle[1] -= 0.05
        if keys[pygame.K_RIGHT]: angle[1] += 0.05
        if keys[pygame.K_UP]: angle[0] -= 0.05
        if keys[pygame.K_DOWN]: angle[0] += 0.05
        if keys[pygame.K_q]: zoom = max(1.0, zoom - 0.1)
        if keys[pygame.K_w]: zoom = min(10.0, zoom + 0.1)

        transformed = [rotate(v, angle) for v in vertices]
        projected = [project(v, zoom) for v in transformed]

        if mode == 'points':
            for x, y, _ in projected:
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    screen.set_at((x, y), WHITE)

        elif mode == 'wire':
            for face in faces:
                for i in range(3):
                    a = projected[face[i]]
                    b = projected[face[(i+1)%3]]
                    pygame.draw.line(screen, WHITE, (a[0], a[1]), (b[0], b[1]), 1)

        elif mode == 'solid':
            for idx, face in enumerate(faces):
                pts = [projected[i] for i in face]
                xy_pts = [(int(p[0]), int(p[1])) for p in pts]
                z_vals = [p[2] for p in pts]
                draw_triangle(screen, zbuffer, xy_pts, z_vals, triangle_colors[idx])

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
