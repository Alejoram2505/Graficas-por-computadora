import glfw
from OpenGL.GL import *
import numpy as np
from PIL import Image
import pyrr
import time
import ctypes

# ---------------- SHADER UTILS ----------------
def compile_shader(path, shader_type):
    with open(path, 'r') as f:
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
    v, vt, vn, faces = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '): v.append(list(map(float, line.split()[1:])))
            elif line.startswith('vt '): vt.append(list(map(float, line.split()[1:])))
            elif line.startswith('vn '): vn.append(list(map(float, line.split()[1:])))
            elif line.startswith('f '):
                tri = []
                for token in line.split()[1:]:
                    vi, ti, ni = (int(i) - 1 for i in token.split('/'))
                    tri.append((vi, ti, ni))
                faces.append(tri)
    data = []
    for face in faces:
        for vi, ti, ni in face:
            data.extend(v[vi]); data.extend(vt[ti]); data.extend(vn[ni])
    return np.array(data, dtype=np.float32)

# ---------------- TEXTURE ----------------
def load_texture(path):
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return tex

# ---------------- MAIN ----------------
def main():
    if not glfw.init():
        raise Exception("Error al inicializar GLFW")

    window = glfw.create_window(900, 800, "Casa Madera - Shaders combinables", None, None)
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.06, 0.08, 1.0)

    # Modelo
    model_data = load_obj("casa_madera.obj")
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

    # Textura base
    diffuse_tex = load_texture("casa_madera.png")

    # Shader combinado
    program = create_program("combo_vertex.glsl", "combo_fragment.glsl")

    # Matrices base
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, 900/700, 0.1, 100.0)
    view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 2.5, 8]),
                                        pyrr.Vector3([0, 1, 0]),
                                        pyrr.Vector3([0, 1, 0]))
    model = pyrr.matrix44.create_identity()
    start_time = time.time()

    # Estados de los efectos
    wave_enabled = False
    glow_enabled = False

    # Controles de teclado
    def key_callback(window, key, scancode, action, mods):
        nonlocal wave_enabled, glow_enabled
        if action == glfw.PRESS:
            if key == glfw.KEY_0:
                wave_enabled = glow_enabled = False
                print("→ Shaders desactivados (modo base)")
            elif key == glfw.KEY_1:
                wave_enabled = not wave_enabled
                print("→ Wave Shader:", "ON" if wave_enabled else "OFF")
            elif key == glfw.KEY_2:
                glow_enabled = not glow_enabled
                print("→ Glow Shader:", "ON" if glow_enabled else "OFF")
    glfw.set_key_callback(window, key_callback)

    # Render loop
    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(program)

        # Tiempo y matrices
        t = time.time() - start_time
        glUniform1f(glGetUniformLocation(program, "u_time"), t)
        glUniformMatrix4fv(glGetUniformLocation(program, "u_model"), 1, GL_FALSE, model)
        glUniformMatrix4fv(glGetUniformLocation(program, "u_view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(program, "u_projection"), 1, GL_FALSE, projection)

        # Luz cálida y ambiente general
        glUniform3f(glGetUniformLocation(program, "u_lightPos"), -2.0, 3.2, 6.0)
        glUniform3f(glGetUniformLocation(program, "u_lightColor"), 1.0, 0.95, 0.9)
        glUniform3f(glGetUniformLocation(program, "u_viewPos"), 0.0, 2.0, 6.0)

        # Parámetros de wave
        glUniform1f(glGetUniformLocation(program, "u_amplitude"), 0.15)
        glUniform1f(glGetUniformLocation(program, "u_frequency"), 2.0)
        glUniform1f(glGetUniformLocation(program, "u_speed"), 1.5)
        glUniform1i(glGetUniformLocation(program, "u_waveEnabled"), wave_enabled)

        # Parámetros de glow
        glUniform1f(glGetUniformLocation(program, "u_glowIntensity"), 2.0)
        glUniform1i(glGetUniformLocation(program, "u_glowEnabled"), glow_enabled)

        # Textura
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, diffuse_tex)
        glUniform1i(glGetUniformLocation(program, "u_diffuseMap"), 0)

        # Dibujar
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, len(model_data)//8)
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
