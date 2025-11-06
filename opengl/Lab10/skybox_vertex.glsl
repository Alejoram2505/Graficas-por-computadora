#version 330 core
layout(location = 0) in vec3 aPos;
out vec3 vTexCoord;
uniform mat4 u_view;
uniform mat4 u_projection;

void main() {
    vTexCoord = aPos;
    mat4 viewNoTrans = mat4(mat3(u_view));     // quitar traslación
    vec4 pos = u_projection * viewNoTrans * vec4(aPos, 1.0);
    gl_Position = pos.xyww;                    // empuja al fondo
}
