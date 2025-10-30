#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

uniform bool u_twistEnabled;
uniform float u_time;
uniform float u_twistAmount;   // cuánto torsiona (radianes por unidad)
uniform float u_pivotY;        // eje base de torsión

out vec3 vNormal;
out vec3 vWorldPos;
out vec2 vTexCoord;

void main()
{
    vec3 pos = aPos;

    if (u_twistEnabled) {
        float angle = (pos.y - u_pivotY) * u_twistAmount;
        float s = sin(angle + u_time * 0.5);
        float c = cos(angle + u_time * 0.5);
        // rotación en torno al eje Y
        mat2 rot = mat2(c, -s, s, c);
        pos.xz = rot * pos.xz;
    }

    vec4 worldPos = u_model * vec4(pos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(u_model))) * aNormal;
    vTexCoord = aTexCoord;

    gl_Position = u_projection * u_view * worldPos;
}
