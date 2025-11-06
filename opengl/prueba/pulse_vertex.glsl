#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

uniform float u_time;
uniform bool  u_pulseEnabled;
uniform float u_pulseSpeed;
uniform float u_pulseAmount;

out vec3 vNormal;
out vec3 vWorldPos;
out vec2 vTexCoord;

void main()
{
    vec3 pos = aPos;

    // --- Pulse (expansión/contracción orgánica) ---
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
