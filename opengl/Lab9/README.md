# README — Lab9 Shaders (OpenGL + GLSL)

## Proyecto
**Laboratorio 9 — Shaders combinables en OpenGL con GLSL**  
El objetivo es implementar y combinar múltiples efectos visuales (vertex y fragment shaders) aplicados sobre un modelo 3D (`casa_madera.obj`) utilizando un renderer en OpenGL.

---

## Modelo y Texturas
- **Modelo:** `casa_madera.obj`  
- **Textura base:** `casa_madera.png`  
- Se renderiza con cámara en perspectiva y luz cálida direccional.  

---

## Shaders implementados

### Vertex Shaders
1. **Wave Shader**   
   - Efecto de ondulación tipo viento.  
   - Desplaza los vértices en eje Y con funciones seno y coseno.  
   - Controlado por tiempo (`u_time`) y parámetros de frecuencia y amplitud.

2. **Twist Shader** 
   - Retuerce el modelo alrededor del eje Y, creando un efecto de torsión.  
   - El giro varía según la altura del vértice y se anima con el tiempo.  

3. **Pulse Shader**  
   - Escala el modelo en todas las direcciones simulando un “latido” o expansión rítmica.  
   - Efecto orgánico tipo respiración o energía.  
   - Ideal combinado con el shader de Lava Cracks.  

---

### Fragment Shaders
1. **Glow Shader**  
   - Añade un brillo cálido en el contorno del modelo.  
   - El borde parpadea con el tiempo, simulando iluminación ambiental.  

2. **Lava Cracks Shader**  
   - Simula grietas incandescentes animadas sobre la superficie del modelo.  
   - El patrón de grietas se mueve dinámicamente con el tiempo.  
   - Incluye un leve efecto “emissive” para dar brillo a las zonas calientes.  

---

## ⚙️ Controles de Teclado
| Tecla | Shader | Tipo | Descripción |
|-------|---------|------|--------------|
| **0** | — | — | Desactiva todos los efectos |
| **1** | Wave | Vertex | Ondulación tipo viento |
| **2** | Glow | Fragment | Brillo cálido parpadeante |
| **3** | Twist | Vertex | Torsión vertical animada |
| **4** | Pulse | Vertex | Expansión/contracción del modelo |
| **5** | Lava Cracks | Fragment | Grietas de lava incandescente |

> Todos los efectos son **combinables** entre sí.  
> Por ejemplo: activar **Wave + Glow + Lava** crea un efecto energético muy visual.

---

## Parámetros personalizables
En el código (`Lab9.py`), puedes ajustar:
```python
# Wave
u_amplitude = 0.15
u_frequency = 2.0
u_speed = 1.5

# Twist
u_twistAmount = 0.5

# Pulse
u_pulseSpeed = 2.0
u_pulseAmount = 0.03

# Lava Cracks
u_crackScale = 10.0
u_crackSpeed = 2.0
u_fireColor = (1.0, 0.45, 0.1)
```

---

## Resultado

![Demostración](lab9.mp4)

---

## Creado por: Diego Ramírez  

