#version 330 core
in vec2 vTexCoord;
in vec3 vNormal;
in vec3 vFragPos;

out vec4 FragColor;

uniform sampler2D u_diffuseMap;
uniform vec3 u_lightPos;
uniform vec3 u_lightColor;
uniform vec3 u_viewPos;
uniform float u_time;

// toggles
uniform int u_lavaEnabled;
uniform int u_glowEnabled;
uniform int u_snowEnabled;

// iluminación básica
vec3 basicLighting(vec3 baseColor, vec3 normal, vec3 fragPos)
{
    vec3 lightDir = normalize(u_lightPos - fragPos);
    vec3 viewDir  = normalize(u_viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, normal);

    float diff = max(dot(normal, lightDir), 0.0);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

    vec3 ambient  = 0.25 * baseColor;
    vec3 diffuse  = diff * baseColor * u_lightColor;
    vec3 specular = spec * vec3(0.25);

    return ambient + diffuse + specular;
}

// Lava cracks procedural
vec3 lavaCracks(vec2 uv, float time, vec3 baseColor)
{
    float s = 7.0;
    uv *= s;
    float n = sin(uv.x + time * 1.8) *
              cos(uv.y * 1.3 - time * 1.2);

    float cracks = smoothstep(0.15, 0.05, abs(n));
    vec3 fireColor = vec3(1.0, 0.45, 0.1);
    vec3 lava = mix(baseColor * 0.3, fireColor, cracks);
    return lava;
}

// Snow overlay
vec3 snowEffect(vec3 color, vec3 normal, vec3 fragPos)
{
    float up = max(dot(normalize(normal), vec3(0.0, 1.0, 0.0)), 0.0);
    float amount = smoothstep(0.4, 0.9, up);
    vec3 snow = vec3(0.92, 0.95, 1.0);
    return mix(color, snow, amount * 0.9);
}

// Glow en bordes
vec3 glowRim(vec3 color, vec3 normal, vec3 viewDir)
{
    float rim = 1.0 - max(dot(normalize(normal), normalize(viewDir)), 0.0);
    rim = pow(rim, 3.0);
    vec3 glowColor = vec3(1.0, 0.8, 0.2);
    return color + glowColor * rim * 1.4;
}

void main()
{
    vec3 baseColor = texture(u_diffuseMap, vTexCoord).rgb;
    vec3 N = normalize(vNormal);

    vec3 lit = basicLighting(baseColor, N, vFragPos);

    if (u_lavaEnabled == 1) {
        lit = lavaCracks(vTexCoord, u_time, lit);
    }

    if (u_snowEnabled == 1) {
        lit = snowEffect(lit, N, vFragPos);
    }

    if (u_glowEnabled == 1) {
        vec3 viewDir = normalize(u_viewPos - vFragPos);
        lit = glowRim(lit, N, viewDir);
    }

    FragColor = vec4(lit, 1.0);
}
