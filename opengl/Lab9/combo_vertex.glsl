#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

uniform float u_time;
uniform float u_amplitude;
uniform float u_frequency;
uniform float u_speed;
uniform bool u_waveEnabled;

out vec3 vNormal;
out vec3 vWorldPos;
out vec2 vTexCoord;

void main()
{
    vec3 pos = aPos;

    // Efecto de ondulación si está activado
    if (u_waveEnabled) {
        float wave = sin((aPos.x + aPos.y) * u_frequency + u_time * u_speed)
                * cos((aPos.z + aPos.y) * u_frequency + u_time * u_speed)
                * u_amplitude;
        pos.y += wave;
    }

    vec4 worldPos = u_model * vec4(pos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(u_model))) * aNormal;
    vTexCoord = aTexCoord;

    gl_Position = u_projection * u_view * worldPos;
}
