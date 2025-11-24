# Proyecto Final — Pueblito Campestre
El objetivo es implementar un visualizador de modelos 3D en OpenGL que permita:
- Carga de 5 modelos OBJ (triangulados, texturizados)
- Cada modelo con su textura y shaders especiales activables por teclado
- Skybox HDR realista usando un .hdr de campo
- Cámara orbital, zoom, y desplazamientos verticales/horizontales → Funciona con mouse y teclado
- Post-procesado en pantalla completa
- Fisheye / Distorsión de lente
- Aberración cromática
- Música ambiental de fondo con pygame, en bucle
- Un piso 3D texturizado con pasto para integrar la escena
- Diseño tipo “Pueblito Campestre”, con una composición estética y coherente

---

## Modelos incluidos en el Diorama

- Planta decorativa
- Casita pequeña
- Casa de madera
- Casa nevada
- Hacha clavada en el suelo
- Piso texturizado de pasto (extra)
---

## Controles del Usuario

- Mouse (mover horizontal y vertical la cámara) 
- Teclado — Cámara
    - Zoom IN	A
    - Zoom OUT	S
    - Orbitar izquierda	←
    - Orbitar derecha	→
    - Subir cámara	↑
    - Bajar cámara	↓
- Cambiar punto de enfoque (modelo que mira la cámara)
    - Planta	1
    - Casita	2
    - Casa de madera	3
    - Casa nevada	4
    - Hacha	5
- Shaders por modelo (activación individual)
    - 6	Wave
    - 7	Pulse
    - 8	Twist
    - 9	Lava cracks
    - 0	Glow
    - N	Nieve (Snow)
    - M	Apagar todos los efectos del modelo
- Post-procesado (Fisheye, Aberración cromática)
    - P	Activar / desactivar postprocesado
    - I	Aumentar fuerza del efecto
    - O	Disminuir fuerza del efecto
- Música Ambiental
    - fondo.mp3 (musica de banjo)

---

## video demostración
![Demostración](https://uvggt-my.sharepoint.com/:v:/g/personal/ram23601_uvg_edu_gt/IQBgfU2OiKvrSp6urgypQfNEAYt61GakQZ2dHUiQo_C7fNw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=bYeWe2)

---

## Creado por: Diego Ramírez
