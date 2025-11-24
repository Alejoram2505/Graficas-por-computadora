#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;

out vec2 vTexCoord;
out vec3 vNormal;
out vec3 vFragPos;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_time;

// toggles
uniform int u_waveEnabled;
uniform int u_pulseEnabled;
uniform int u_twistEnabled;

// parámetros
const float WAVE_AMPLITUDE = 0.25;
const float WAVE_FREQ = 2.2;
const float WAVE_SPEED = 1.3;

const float PULSE_SPEED = 2.0;
const float TWIST_STRENGTH = 0.6;

void main()
{
    vec3 pos = aPos;

    // Wave (tipo viento en Y usando XZ)
    if (u_waveEnabled == 1) {
        float w = sin(pos.x * WAVE_FREQ + u_time * WAVE_SPEED)
                + cos(pos.z * WAVE_FREQ * 0.7 + u_time * (WAVE_SPEED * 1.3));
        pos.y += WAVE_AMPLITUDE * w * 0.5;
    }

    // Pulse (escala radial desde el centro)
    if (u_pulseEnabled == 1) {
        float s = 1.0 + 0.08 * sin(u_time * PULSE_SPEED);
        pos.xy *= s;
    }

    // Twist alrededor de Y
    if (u_twistEnabled == 1) {
        float angle = TWIST_STRENGTH * pos.y;
        float c = cos(angle);
        float s = sin(angle);
        mat2 rot = mat2(c, -s,
                        s,  c);
        pos.xz = rot * pos.xz;
    }

    vec4 worldPos = u_model * vec4(pos, 1.0);
    vFragPos = worldPos.xyz;

    mat3 normalMatrix = mat3(transpose(inverse(u_model)));
    vNormal = normalize(normalMatrix * aNormal);

    vTexCoord = aTexCoord;
    gl_Position = u_projection * u_view * worldPos;
}
