#version 330 core

in vec3 vNormal;
in vec3 vWorldPos;
in vec2 vTexCoord;

out vec4 FragColor;

uniform sampler2D u_diffuseMap;
uniform vec3 u_lightPos;
uniform vec3 u_lightColor;
uniform vec3 u_viewPos;
uniform float u_time;

uniform bool  u_glowEnabled;
uniform float u_glowIntensity;

uniform bool  u_lavaEnabled;
uniform float u_crackScale;
uniform float u_crackSpeed;
uniform vec3  u_fireColor;

void main()
{
    vec3 baseColor = texture(u_diffuseMap, vTexCoord).rgb;
    vec3 N = normalize(vNormal);
    vec3 L = normalize(u_lightPos - vWorldPos);
    vec3 V = normalize(u_viewPos - vWorldPos);

    float diff = max(dot(N, L), 0.0);
    vec3 ambient = 0.4 * baseColor;
    vec3 diffuse = diff * baseColor * u_lightColor;
    vec3 finalColor = ambient + diffuse;

    // --- Glow (contorno cálido) ---
    if (u_glowEnabled) {
        float flicker = 0.5 + 0.5 * sin(u_time * 3.0);
        float edge = pow(1.0 - abs(dot(N, V)), 3.0);
        vec3 glowColor = vec3(1.0, 0.9, 0.6) * edge * flicker * u_glowIntensity;
        finalColor += glowColor;
    }

    // --- Lava Cracks ---
    if (u_lavaEnabled) {
        vec2 uv = vTexCoord * u_crackScale;
        uv += vec2(sin(u_time * u_crackSpeed) * 0.2, cos(u_time * u_crackSpeed * 0.7) * 0.2);

        float cracks = sin(uv.x + uv.y + sin(uv.x * 1.5) + cos(uv.y * 1.5));
        cracks = abs(fract(cracks) - 0.5) * 2.0;

        float intensity = smoothstep(0.3, 0.7, cracks);
        vec3 lava = mix(u_fireColor, baseColor, intensity);
        finalColor = mix(finalColor, lava, 0.8);
    }

    FragColor = vec4(finalColor, 1.0);
}
