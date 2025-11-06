import glfw
from OpenGL.GL import *
import numpy as np
from PIL import Image
import pyrr
import math, time, ctypes
import imageio.v3 as iio

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

# ---------------- OBJ LOADER ----------------
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
                # posición
                if 0 <= vi < len(positions):
                    data.extend(positions[vi])
                else:
                    data.extend([0.0, 0.0, 0.0])
                # uv
                if 0 <= ti < len(texcoords):
                    data.extend(texcoords[ti])
                else:
                    data.extend([0.0, 0.0])
                # normal
                if 0 <= ni < len(normals):
                    data.extend(normals[ni])
                else:
                    data.extend([0.0, 1.0, 0.0])

    return np.array(data, dtype=np.float32)

# ---------------- TEXTURE LOADER ----------------
def load_texture(path):
    if path.lower().endswith(".hdr"):
        img = iio.imread(path).astype(np.float32)
        img = np.clip(img / np.max(img), 0, 1.0)
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tex

# ---------------- Fullscreen Triangle  ----------------
def create_fullscreen_triangle():
    verts = np.array([
        -1.0, -1.0,
         3.0, -1.0,
        -1.0,  3.0,
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
        self.radius = 8.0
        self.theta = 0.0
        self.phi = 0.3
        self.target = pyrr.Vector3([0, 1.5, 0])
        self.zoom_speed = 0.8
        self.mouse_sensitivity = 0.015
        self.vertical_limit = [0.1, 1.3]

    def get_view(self):
        cam_x = self.radius * math.sin(self.theta) * math.cos(self.phi)
        cam_y = self.radius * math.sin(self.phi)
        cam_z = self.radius * math.cos(self.theta) * math.cos(self.phi)
        pos = pyrr.Vector3([cam_x, cam_y, cam_z]) + self.target
        return pyrr.matrix44.create_look_at(pos, self.target, pyrr.Vector3([0, 1, 0])), pos

# ---------------- MAIN ----------------
def main():
    if not glfw.init():
        raise Exception("Error al inicializar GLFW")

    window = glfw.create_window(1000, 800, "Lab10 - Visualizador 3D", None, None)
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glClearColor(0.05, 0.06, 0.08, 1.0)

    # Modelos y texturas
    models = [
        ("Planta.obj", "planta.jpg"),
        ("Axe.obj", "Axe_1.png"),
        ("casa_madera.obj", "casa_madera.png")
    ]
    vao_list, tex_list, size_list = [], [], []
    for obj_path, tex_path in models:
        model_data = load_obj(obj_path)
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, model_data.nbytes, model_data, GL_STATIC_DRAW)
        stride = 8 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)
        vao_list.append(vao)
        tex_list.append(load_texture(tex_path))
        size_list.append(len(model_data)//8)

    current_model = 0
    scales = [3.0, 0.01, 1.0]  # Planta, Hacha, Casa

    # Shaders
    model_program  = create_program("combo_vertex.glsl",  "combo_fragment.glsl")
    skybox_program = create_program("skybox_fullscreen.vert", "skybox_fullscreen.frag")

    # Skybox 
    fs_vao = create_fullscreen_triangle()
    hdr_tex = load_texture("metro.hdr")

    # Cámara
    cam = OrbitCamera()
    last_x, last_y, rotating = 0, 0, False

    # Estados efectos
    wave_enabled = glow_enabled = twist_enabled = pulse_enabled = lava_enabled = False

    # Input
    def key_callback(window, key, scancode, action, mods):
        nonlocal current_model, wave_enabled, glow_enabled, twist_enabled, pulse_enabled, lava_enabled
        if action == glfw.PRESS:
            if key == glfw.KEY_1: current_model = 0
            elif key == glfw.KEY_2: current_model = 1
            elif key == glfw.KEY_3: current_model = 2
            elif key == glfw.KEY_4: wave_enabled  = not wave_enabled
            elif key == glfw.KEY_5: glow_enabled  = not glow_enabled
            elif key == glfw.KEY_6: twist_enabled = not twist_enabled
            elif key == glfw.KEY_7: pulse_enabled = not pulse_enabled
            elif key == glfw.KEY_8: lava_enabled  = not lava_enabled
            elif key == glfw.KEY_9:
                wave_enabled = glow_enabled = twist_enabled = pulse_enabled = lava_enabled = False
            elif key == glfw.KEY_Q: cam.radius = max(3.0,  cam.radius - cam.zoom_speed)
            elif key == glfw.KEY_E: cam.radius = min(15.0, cam.radius + cam.zoom_speed)
    glfw.set_key_callback(window, key_callback)

    def cursor_pos(window, xpos, ypos):
        nonlocal last_x, last_y, rotating
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

    start_time = time.time()

    # -------- MAIN LOOP --------
    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        t = time.time() - start_time

        projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1000/800, 0.1, 100.0)
        view, cam_pos = cam.get_view()

        # ===== SKYBOX (fullscreen) =====
        glDepthMask(GL_FALSE)            
        glDepthFunc(GL_LEQUAL)           
        glUseProgram(skybox_program)

        # mandamos inversas para reconstruir el rayo
        invProj = np.linalg.inv(projection).astype(np.float32)
        view_rot = np.array(pyrr.matrix33.create_from_matrix44(view)).astype(np.float32)
        invViewRot = np.linalg.inv(view_rot).astype(np.float32)

        glUniformMatrix4fv(glGetUniformLocation(skybox_program, "u_invProj"), 1, GL_FALSE, invProj)
        glUniformMatrix3fv(glGetUniformLocation(skybox_program, "u_invViewRot"), 1, GL_FALSE, invViewRot)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, hdr_tex)
        glUniform1i(glGetUniformLocation(skybox_program, "u_envMap"), 0)

        glBindVertexArray(fs_vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)

        # ===== MODEL =====
        glUseProgram(model_program)
        scale_factor = scales[current_model]
        model_matrix = pyrr.matrix44.create_from_scale(pyrr.Vector3([scale_factor]*3))
        glUniformMatrix4fv(glGetUniformLocation(model_program, "u_model"), 1, GL_FALSE, model_matrix)
        glUniformMatrix4fv(glGetUniformLocation(model_program, "u_view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(model_program, "u_projection"), 1, GL_FALSE, projection)
        glUniform1f(glGetUniformLocation(model_program, "u_time"), t)
        glUniform3f(glGetUniformLocation(model_program, "u_lightPos"), -2.0, 3.2, 6.0)
        glUniform3f(glGetUniformLocation(model_program, "u_lightColor"), 1.0, 0.95, 0.9)
        glUniform3f(glGetUniformLocation(model_program, "u_viewPos"), *cam_pos)

        # toggles + parámetros (adaptados a la escala)
        glUniform1i(glGetUniformLocation(model_program, "u_waveEnabled"),  wave_enabled)
        glUniform1i(glGetUniformLocation(model_program, "u_glowEnabled"),  glow_enabled)
        glUniform1i(glGetUniformLocation(model_program, "u_twistEnabled"), twist_enabled)
        glUniform1i(glGetUniformLocation(model_program, "u_pulseEnabled"), pulse_enabled)
        glUniform1i(glGetUniformLocation(model_program, "u_lavaEnabled"),  lava_enabled)

        glUniform1f(glGetUniformLocation(model_program, "u_amplitude"),     0.15 * scale_factor)
        glUniform1f(glGetUniformLocation(model_program, "u_frequency"),     2.0)
        glUniform1f(glGetUniformLocation(model_program, "u_speed"),         1.5)
        glUniform1f(glGetUniformLocation(model_program, "u_glowIntensity"), 2.0)
        glUniform1f(glGetUniformLocation(model_program, "u_twistAmount"),   max(0.05, 0.5 / max(0.01, scale_factor)))
        glUniform1f(glGetUniformLocation(model_program, "u_pulseSpeed"),    2.0)
        glUniform1f(glGetUniformLocation(model_program, "u_pulseAmount"),   max(0.005, 0.03 / max(0.01, scale_factor)))
        glUniform1f(glGetUniformLocation(model_program, "u_crackScale"),    10.0)
        glUniform1f(glGetUniformLocation(model_program, "u_crackSpeed"),    2.0)
        glUniform3f(glGetUniformLocation(model_program, "u_fireColor"),     1.0, 0.45, 0.1)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_list[current_model])
        glUniform1i(glGetUniformLocation(model_program, "u_diffuseMap"), 0)

        glBindVertexArray(vao_list[current_model])
        glDrawArrays(GL_TRIANGLES, 0, size_list[current_model])

        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
