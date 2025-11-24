import glfw
from OpenGL.GL import *
import numpy as np
from PIL import Image
import pyrr
import math, time, ctypes
import imageio.v3 as iio
import pygame

# ---------------- SHADER UTILS ----------------
def compile_shader(path, shader_type):
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def create_program(vs_path, fs_path):
    vs = compile_shader(vs_path, GL_VERTEX_SHADER)
    fs = compile_shader(fs_path, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(program).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    print(f"[OK] Shaders vinculados: {vs_path} + {fs_path}")
    return program

# ---------------- OBJ LOADER (triangulación) ----------------
def load_obj(path):
    positions, texcoords, normals = [], [], []
    faces = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line or line[0] == '#':
                continue
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':
                positions.append(list(map(float, parts[1:4])))
            elif parts[0] == 'vt':
                texcoords.append(list(map(float, parts[1:3])))
            elif parts[0] == 'vn':
                normals.append(list(map(float, parts[1:4])))
            elif parts[0] == 'f':
                face = []
                for tok in parts[1:]:
                    vtn = tok.split('/')
                    vi = int(vtn[0]) - 1 if vtn[0] else -1
                    ti = int(vtn[1]) - 1 if len(vtn) > 1 and vtn[1] else -1
                    ni = int(vtn[2]) - 1 if len(vtn) > 2 and vtn[2] else -1
                    face.append((vi, ti, ni))
                faces.append(face)

    data = []
    for face in faces:
        if len(face) < 3:
            continue
        v0 = face[0]
        for i in range(1, len(face)-1):
            tri = (v0, face[i], face[i+1])
            for (vi, ti, ni) in tri:
                if 0 <= vi < len(positions):
                    data.extend(positions[vi])
                else:
                    data.extend([0.0, 0.0, 0.0])

                if 0 <= ti < len(texcoords):
                    data.extend(texcoords[ti])
                else:
                    data.extend([0.0, 0.0])

                if 0 <= ni < len(normals):
                    data.extend(normals[ni])
                else:
                    data.extend([0.0, 1.0, 0.0])

    return np.array(data, dtype=np.float32)

# ---------------- TEXTURE LOADER ----------------
def load_texture(path):
    if path.lower().endswith(".hdr"):
        img = iio.imread(path).astype(np.float32)
        img = np.flipud(img)
        # Normalizamos a 0-1 para que funcione como skymap y no queme la escena
        maxv = np.max(img)
        if maxv > 0:
            img = img / maxv
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F,
                     img.shape[1], img.shape[0], 0,
                     GL_RGB, GL_FLOAT, img)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tex
    else:
        img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     img.width, img.height, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tex

# ---------------- FULLSCREEN QUAD ----------------
def create_fullscreen_quad():
    # 2D quad en NDC
    verts = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    return vao

# ---------------- CAMERA ORBIT ----------------
class OrbitCamera:
    def __init__(self):
        self.radius = 15.0
        self.theta = 0.0
        self.phi = 0.35
        self.target = pyrr.Vector3([0, 2.0, 0])
        self.zoom_speed = 0.8
        self.mouse_sensitivity = 0.015
        self.vertical_limit = [0.1, 1.4]

    def get_view(self):
        cam_x = self.radius * math.sin(self.theta) * math.cos(self.phi)
        cam_y = self.radius * math.sin(self.phi)
        cam_z = self.radius * math.cos(self.theta) * math.cos(self.phi)
        pos = pyrr.Vector3([cam_x, cam_y, cam_z]) + self.target
        view = pyrr.matrix44.create_look_at(pos, self.target, pyrr.Vector3([0, 1, 0]))
        return view, pos

# ---------------- MAIN ----------------
def main():
    if not glfw.init():
        raise Exception("Error al inicializar GLFW")

    width, height = 1280, 720
    window = glfw.create_window(width, height, "Proyecto Final - Pueblito Campestre", None, None)
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.06, 0.08, 1.0)

    # --- Música con pygame ---
    pygame.mixer.init()
    try:
        pygame.mixer.music.load("fondo.mp3")
        pygame.mixer.music.set_volume(0.4)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print("No se pudo cargar música:", e)

    # --- Shaders ---
    model_program  = create_program("shaders/combo_vertex.glsl",  "shaders/combo_fragment.glsl")
    skybox_program = create_program("shaders/skybox_fullscreen.vert", "shaders/skybox_fullscreen.frag")
    post_program   = create_program("shaders/postprocess_vertex.glsl", "shaders/postprocess_fisheye.frag")

    # --- Models ---
    model_files = [
        ("objetos/Planta.obj",      "texturas/planta.jpg"),     # 0
        ("objetos/casita.obj",      "texturas/casita.png"),     # 1
        ("objetos/casa_madera.obj", "texturas/casa_madera.png"),# 2
        ("objetos/casa_nieve.obj",  "texturas/casa_nieve.jpg"), # 3
        ("objetos/Axe.obj",         "texturas/Axe.png"),        # 4 (ajusta nombre si es distinto)
    ]

    vao_list, tex_list, size_list = [], [], []
    for obj_path, tex_path in model_files:
        data = load_obj(obj_path)
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        stride = 8 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)

        vao_list.append(vao)
        tex_list.append(load_texture(tex_path))
        size_list.append(len(data)//8)

    # Posiciones y escalas del diorama (Pueblito Campestre)
    # 0 Planta, 1 Casita, 2 Casa madera, 3 Casa nieve, 4 Hacha
    positions = [
        pyrr.Vector3([ -6.0,  0.0,  3.0 ]),   # Planta (frente izquierda)
        pyrr.Vector3([ -8.0,  0.0, -2.0 ]),   # Casita pequeña (izquierda atrás)
        pyrr.Vector3([  0.0,  0.0,  0.0 ]),   # Casa madera (centro del pueblo)
        pyrr.Vector3([  5.0,  0.0, -3.0 ]),   # Casa nieve (derecha atrás)
        pyrr.Vector3([  1.5,  0.0,  3.5 ]),   # Hacha (frente derecha)
    ]

    scales = [
        1.0,   # Planta
        0.8,   # Casita
        2.0,   # Casa madera
        0.1,   # Casa nieve
        0.01,  # Hacha (scaneo gigante)
    ]

    # --- Skybox ---
    hdr_tex = load_texture("Campo.hdr")
    fs_quad_sky = create_fullscreen_quad()

    # --- Post-process ---
    fs_quad_post = fs_quad_sky  # podemos reutilizar el mismo quad

    # Framebuffer para post-processing
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    color_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, color_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0)

    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo)

    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("FBO incompleto!")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # --- Cámara ---
    cam = OrbitCamera()
    last_x, last_y = 0, 0
    rotating = False

    # Fisheye / post
    post_enabled = False 
    fisheye_strength = 0.28  # se controla con I / O

    # Punto focal (índice de modelo)
    focus_index = 2  # casa madera

    # Combinaciones de shaders por modelo
    # wave, pulse, twist, lava, glow, snow

    shader_sets = [
        (False, False, False, False, False, False),# Planta: wave + lava cracks
        (False, False, False, False, False, False),# Casita: pulse + twist suave
        (False, False, False, False, False, False),# Casa madera: lava cracks sutil
        (False, False, False, False, False, False),# Casa nieve: wave leve + snow
        (False, False, False, False, False, False),# Hacha: pulse + glow
    ]
   
    # Input callbacks
    def key_callback(window, key, scancode, action, mods):
        nonlocal cam, post_enabled, fisheye_strength, focus_index
        if action == glfw.PRESS or action == glfw.REPEAT:
            # Cambio de foco de cámara
            if key == glfw.KEY_1: focus_index = 0
            elif key == glfw.KEY_2: focus_index = 1
            elif key == glfw.KEY_3: focus_index = 2
            elif key == glfw.KEY_4: focus_index = 3
            elif key == glfw.KEY_5: focus_index = 4

            # Zoom con teclado (A/S)
            elif key == glfw.KEY_A:
                cam.radius = max(5.0, cam.radius - cam.zoom_speed)
            elif key == glfw.KEY_S:
                cam.radius = min(30.0, cam.radius + cam.zoom_speed)

            # Movimiento orbital con flechas
            elif key == glfw.KEY_LEFT:
                cam.theta -= 0.05
            elif key == glfw.KEY_RIGHT:
                cam.theta += 0.05
            elif key == glfw.KEY_UP:
                cam.phi = min(cam.vertical_limit[1], cam.phi + 0.05)
            elif key == glfw.KEY_DOWN:
                cam.phi = max(cam.vertical_limit[0], cam.phi - 0.05)

            # Toggle postprocess
            elif key == glfw.KEY_P:
                post_enabled = not post_enabled

            # Intensidad fisheye (I/O)
            elif key == glfw.KEY_I:
                fisheye_strength = min(1.0, fisheye_strength + 0.05)
            elif key == glfw.KEY_O:
                fisheye_strength = max(0.0, fisheye_strength - 0.05)

            # Activación manual de shaders por tecla 
            elif key == glfw.KEY_6:   # Wave ON/OFF
                w, p, t, l, g, s = shader_sets[focus_index]
                shader_sets[focus_index] = (not w, p, t, l, g, s)

            elif key == glfw.KEY_7:   # Pulse ON/OFF
                w, p, t, l, g, s = shader_sets[focus_index]
                shader_sets[focus_index] = (w, not p, t, l, g, s)

            elif key == glfw.KEY_8:   # Twist ON/OFF
                w, p, t, l, g, s = shader_sets[focus_index]
                shader_sets[focus_index] = (w, p, not t, l, g, s)

            elif key == glfw.KEY_9:   # Lava Cracks ON/OFF
                w, p, t, l, g, s = shader_sets[focus_index]
                shader_sets[focus_index] = (w, p, t, not l, g, s)

            elif key == glfw.KEY_0:   # Glow ON/OFF
                w, p, t, l, g, s = shader_sets[focus_index]
                shader_sets[focus_index] = (w, p, t, l, not g, s)

            elif key == glfw.KEY_N:   # Snow ON/OFF
                w, p, t, l, g, s = shader_sets[focus_index]
                shader_sets[focus_index] = (w, p, t, l, g, not s)

            elif key == glfw.KEY_M:   # Apagar TODO para modelo actual
                shader_sets[focus_index] = (False, False, False, False, False, False)

            

    glfw.set_key_callback(window, key_callback)

    def cursor_pos(window, xpos, ypos):
        nonlocal last_x, last_y, rotating, cam
        if rotating:
            dx, dy = xpos - last_x, ypos - last_y
            cam.theta += dx * cam.mouse_sensitivity
            cam.phi   -= dy * cam.mouse_sensitivity
            cam.phi = max(cam.vertical_limit[0], min(cam.vertical_limit[1], cam.phi))
        last_x, last_y = xpos, ypos

    glfw.set_cursor_pos_callback(window, cursor_pos)

    def mouse_button(window, button, action, mods):
        nonlocal rotating
        if button == glfw.MOUSE_BUTTON_LEFT:
            rotating = (action == glfw.PRESS)

    glfw.set_mouse_button_callback(window, mouse_button)

    # Tiempo
    start_time = time.time()

    # --- Loop principal ---
    while not glfw.window_should_close(window):
        glfw.poll_events()
        t = time.time() - start_time

        # La cámara siempre mira al modelo de focus
        cam.target = positions[focus_index] + pyrr.Vector3([0.0, 2.0, 0.0])
        view, cam_pos = cam.get_view()
        projection = pyrr.matrix44.create_perspective_projection_matrix(45, width/height, 0.1, 200.0)

        # --------- 1) Render a FBO (scene + skybox) ----------
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ---- Skybox fullscreen ----
        glDepthMask(GL_FALSE)
        glUseProgram(skybox_program)

        invProj = np.linalg.inv(projection).astype(np.float32)
        view_rot = np.array(pyrr.matrix33.create_from_matrix44(view)).astype(np.float32)
        invViewRot = np.linalg.inv(view_rot).astype(np.float32)

        glUniformMatrix4fv(glGetUniformLocation(skybox_program, "u_invProj"), 1, GL_FALSE, invProj)
        glUniformMatrix3fv(glGetUniformLocation(skybox_program, "u_invViewRot"), 1, GL_FALSE, invViewRot)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, hdr_tex)
        glUniform1i(glGetUniformLocation(skybox_program, "u_envMap"), 0)

        glBindVertexArray(fs_quad_sky)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDepthMask(GL_TRUE)

        # ---- Modelos ----
        glUseProgram(model_program)
        glUniformMatrix4fv(glGetUniformLocation(model_program, "u_view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(model_program, "u_projection"), 1, GL_FALSE, projection)
        glUniform1f(glGetUniformLocation(model_program, "u_time"), t)
        glUniform3f(glGetUniformLocation(model_program, "u_lightPos"), 4.0, 10.0, 4.0)
        glUniform3f(glGetUniformLocation(model_program, "u_lightColor"), 1.0, 0.97, 0.9)
        glUniform3f(glGetUniformLocation(model_program, "u_viewPos"), cam_pos.x, cam_pos.y, cam_pos.z)

        for i in range(len(vao_list)):
            wave, pulse, twist, lava, glow, snow = shader_sets[i]
            glUniform1i(glGetUniformLocation(model_program, "u_waveEnabled"),  int(wave))
            glUniform1i(glGetUniformLocation(model_program, "u_pulseEnabled"), int(pulse))
            glUniform1i(glGetUniformLocation(model_program, "u_twistEnabled"), int(twist))
            glUniform1i(glGetUniformLocation(model_program, "u_lavaEnabled"),  int(lava))
            glUniform1i(glGetUniformLocation(model_program, "u_glowEnabled"),  int(glow))
            glUniform1i(glGetUniformLocation(model_program, "u_snowEnabled"),  int(snow))

            scale = scales[i]
            model = pyrr.matrix44.create_identity()
            model = pyrr.matrix44.multiply(
                        model,
                        pyrr.matrix44.create_from_scale(pyrr.Vector3([scale, scale, scale]))
                    )
            model = pyrr.matrix44.multiply(
                        model,
                        pyrr.matrix44.create_from_translation(positions[i])
                    )

            glUniformMatrix4fv(glGetUniformLocation(model_program, "u_model"), 1, GL_FALSE, model)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, tex_list[i])
            glUniform1i(glGetUniformLocation(model_program, "u_diffuseMap"), 0)

            glBindVertexArray(vao_list[i])
            glDrawArrays(GL_TRIANGLES, 0, size_list[i])

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # --------- 2) Post-process al backbuffer ----------
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(post_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, color_tex)
        glUniform1i(glGetUniformLocation(post_program, "u_screenTex"), 0)
        glUniform1i(glGetUniformLocation(post_program, "u_enableFisheye"), int(post_enabled))
        glUniform1f(glGetUniformLocation(post_program, "u_strength"), fisheye_strength)

        glBindVertexArray(fs_quad_post)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glfw.swap_buffers(window)

    pygame.mixer.music.stop()
    pygame.mixer.quit()
    glfw.terminate()

if __name__ == "__main__":
    main()
