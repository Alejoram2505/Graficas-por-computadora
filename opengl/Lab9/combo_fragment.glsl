#version 330 core

in vec3 vNormal;
in vec3 vWorldPos;
in vec2 vTexCoord;

out vec4 FragColor;

uniform sampler2D u_diffuseMap;

uniform vec3 u_lightPos;
uniform vec3 u_lightColor;
uniform vec3 u_viewPos;

uniform float u_glowIntensity;
uniform bool u_glowEnabled;
uniform float u_time;

void main()
{
    vec3 baseColor = texture(u_diffuseMap, vTexCoord).rgb;

    // ---- iluminación base ----
    vec3 N = normalize(vNormal);
    vec3 L = normalize(u_lightPos - vWorldPos);
    vec3 V = normalize(u_viewPos - vWorldPos);
    vec3 R = reflect(-L, N);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(V, R), 0.0), 32.0);

    vec3 ambient  = 0.45 * baseColor;
    vec3 diffuse  = diff * baseColor * u_lightColor;
    vec3 specular = spec * vec3(0.9);
    vec3 finalColor = ambient + diffuse + specular;

    // ---- efecto glow parpadeante en bordes ----
    if (u_glowEnabled) {
        float flicker = 0.5 + 0.5 * sin(u_time * 3.0);     // parpadeo
        float edge = pow(1.0 - abs(dot(N, V)), 3.0);       // detecta bordes
        vec3 glowColor = vec3(1.0, 0.9, 0.6) * edge * flicker * u_glowIntensity;
        finalColor += glowColor;
    }

    FragColor = vec4(finalColor, 1.0);
}
