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

    // iluminación simple
    float diff = max(dot(N, L), 0.0);
    vec3 ambient = 0.35 * baseColor;
    vec3 diffuse = diff * baseColor * u_lightColor;
    vec3 finalColor = ambient + diffuse;

    // --- Lava Cracks ---
    if (u_lavaEnabled) {
        vec2 uv = vTexCoord * u_crackScale;
        uv += vec2(sin(u_time * u_crackSpeed) * 0.2, cos(u_time * u_crackSpeed * 0.8) * 0.2);

        // patrón de grietas procedural simple
        float cracks = sin(uv.x * 2.0 + sin(uv.y * 3.0 + u_time * 1.5));
        cracks = abs(fract(cracks * 2.0) - 0.5) * 2.0;

        float intensity = smoothstep(0.3, 0.8, cracks);
        vec3 lava = mix(u_fireColor, baseColor, intensity);

        // brillo autoiluminado (lava incandescente)
        float glow = pow(1.0 - intensity, 2.0);
        lava += u_fireColor * glow * 0.8;

        finalColor = mix(finalColor, lava, 0.9);
    }

    FragColor = vec4(finalColor, 1.0);
}
