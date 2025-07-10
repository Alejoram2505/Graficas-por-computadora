import tkinter as tk

WIDTH = 800
HEIGHT = 600

# Polígonos originales
original_polygons = [
    [(165, 380), (185, 360), (180, 330), (207, 345), (233, 330),
     (230, 360), (250, 380), (220, 385), (205, 410), (193, 383)],

    [(321, 335), (288, 286), (339, 251), (374, 302)],

    [(377, 249), (411, 197), (436, 249)],

    [(413, 177), (448, 159), (502, 88), (553, 53), (535, 36),
     (676, 37), (660, 52), (750, 145), (761, 179), (672, 192),
     (659, 214), (615, 214), (632, 230), (580, 230), (597, 215),
     (552, 214), (517, 144), (466, 180)],

    # Agujero (polígono 5)
    [(682, 175), (708, 120), (735, 148), (739, 170)]
]

# Colores para los polígonos
fill_colors = ['lightblue', 'lightgreen', 'orange', 'gray', 'white']  

# Invertir eje Y
polygons = [
    [(x, HEIGHT - y) for (x, y) in poly]
    for poly in original_polygons
]

holes = [4]

def draw_line(canvas, x0, y0, x1, y1, color='black'):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            canvas.create_line(x, y, x + 1, y, fill=color)
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            canvas.create_line(x, y, x, y + 1, fill=color)
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    canvas.create_line(x, y, x + 1, y, fill=color)

def fill_polygon(canvas, vertices, color='gray'):
    edges = []
    n = len(vertices)
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        if y0 == y1:
            continue
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        edges.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'inv_slope': (x1 - x0)/(y1 - y0)})

    y_min = max(min(p[1] for p in vertices), 0)
    y_max = min(max(p[1] for p in vertices), HEIGHT - 1)

    for y in range(int(y_min), int(y_max)):
        x_intersections = []
        for edge in edges:
            if edge['y0'] <= y < edge['y1']:
                x = edge['x0'] + (y - edge['y0']) * edge['inv_slope']
                x_intersections.append(x)
        x_intersections.sort()
        for i in range(0, len(x_intersections), 2):
            if i+1 < len(x_intersections):
                x_start = int(x_intersections[i])
                x_end = int(x_intersections[i+1])
                canvas.create_line(x_start, y, x_end, y, fill=color)

def draw_polygons(canvas):
    for idx, polygon in enumerate(polygons):
        fill_color = fill_colors[idx] if idx < len(fill_colors) else 'gray'
        fill_polygon(canvas, polygon, color=fill_color)
        for i in range(len(polygon)):
            x0, y0 = polygon[i]
            x1, y1 = polygon[(i + 1) % len(polygon)]
            draw_line(canvas, x0, y0, x1, y1, color='black')

root = tk.Tk()
root.title("Relleno de Polígonos - Colores y Agujeros")
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
canvas.pack()

draw_polygons(canvas)

root.mainloop()
