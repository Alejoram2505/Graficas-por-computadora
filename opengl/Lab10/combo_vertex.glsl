#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

uniform float u_time;
uniform bool  u_waveEnabled;
uniform float u_amplitude;
uniform float u_frequency;
uniform float u_speed;

uniform bool  u_twistEnabled;
uniform float u_twistAmount;
uniform float u_pivotY;

uniform bool  u_pulseEnabled;
uniform float u_pulseSpeed;
uniform float u_pulseAmount;

out vec3 vNormal;
out vec3 vWorldPos;
out vec2 vTexCoord;

void main()
{
    vec3 pos = aPos;

    // --- Wave ---
    if (u_waveEnabled) {
        float wave = sin((aPos.x + aPos.y) * u_frequency + u_time * u_speed)
                   * cos((aPos.z + aPos.y) * u_frequency + u_time * u_speed)
                   * u_amplitude;
        pos.y += wave;
    }

    // --- Twist ---
    if (u_twistEnabled) {
        float angle = (pos.y - u_pivotY) * u_twistAmount;
        float s = sin(angle + u_time * 0.5);
        float c = cos(angle + u_time * 0.5);
        mat2 rot = mat2(c, -s, s, c);
        pos.xz = rot * pos.xz;
    }

    // --- Pulse (expansión/contracción global) ---
    if (u_pulseEnabled) {
        float scale = 1.0 + sin(u_time * u_pulseSpeed) * u_pulseAmount;
        pos *= scale;
    }

    vec4 worldPos = u_model * vec4(pos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(u_model))) * aNormal;
    vTexCoord = aTexCoord;
    gl_Position = u_projection * u_view * worldPos;
}
