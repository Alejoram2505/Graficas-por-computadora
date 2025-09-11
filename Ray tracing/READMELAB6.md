# Ray Tracer - Esferas con Materiales

## 📌 Descripción
Este proyecto implementa un **ray tracer básico en Python** usando `numpy`, `pygame` y soporte para cargar un **environment map** (`.hdr` o `.jpg/.png`).  
El objetivo es renderizar **seis esferas** con diferentes materiales (opacos, reflectivos y transparentes) dentro de un entorno realista.

![Render final](esferas.jpg)

---

## 🎯 Objetivos del laboratorio
- Preparar un ambiente de desarrollo para un modelo de iluminación simple.
- Dibujar esferas con materiales de distintos tipos:
  - **2 opacas**
  - **2 reflectivas**
  - **2 transparentes**
- Cargar un **environment map** como textura de fondo (diferente al usado en clase).
- Renderizar todas las esferas completamente visibles en el framebuffer final.

---

## ⚙️ Funcionamiento del código

### 1. **Ray Tracing**
- Se emite un rayo por cada píxel desde la cámara.
- Se calcula la intersección con las esferas de la escena.
- Para cada impacto se evalúan:
  - **Color local (Phong)**: componentes ambiental, difusa y especular.
  - **Reflexión**: usando la dirección reflejada y recursión.
  - **Transparencia/Refracción**: con el modelo de Snell y Fresnel-Schlick.
- Si un rayo no intersecta ningún objeto, se obtiene el color del **environment map**.

### 2. **Materiales**
Cada esfera tiene parámetros distintos:
- **Opacas**: color difuso dominante.
- **Reflectivas**: reflejan fuertemente el entorno (metal, espejo).
- **Transparentes**: transmiten luz, simulan vidrio con índice de refracción (`ior`).

### 3. **Environment Map**
- Se carga una textura panorámica (`.hdr` o `.jpg/.png`) que representa el mundo exterior.
- Se utiliza un mapeo **equirectangular** para obtener el color correspondiente según la dirección del rayo.
- Implementa **filtrado bilineal** para suavizar el muestreo de la textura.
- Para HDR se aplica un **tone mapping con exposición y gamma** para equilibrar la luminosidad.

### 4. **Render progresivo**
- La imagen se va dibujando **fila por fila** en la ventana, mostrando el progreso del render en tiempo real.

## Creado por Diego Ramírez