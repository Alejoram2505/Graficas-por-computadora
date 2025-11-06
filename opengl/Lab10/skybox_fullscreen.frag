#version 330 core
in vec2 vPosNDC;
out vec4 FragColor;

uniform sampler2D u_envMap;
uniform mat4 u_invProj;
uniform mat3 u_invViewRot;

void main()
{
    // reconstrucción del rayo de cámara
    vec4 ndc = vec4(vPosNDC, 1.0, 1.0);
    vec4 viewDirH = u_invProj * ndc;
    vec3 viewDir = normalize(viewDirH.xyz / viewDirH.w);
    vec3 dir = normalize(u_invViewRot * viewDir);

    // convertir dirección a coordenadas equirectangulares
    float PI = 3.14159265;
    vec2 uv = vec2(
        atan(dir.z, dir.x) / (2.0 * PI) + 0.5,
        asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5
    );

    vec3 hdr = texture(u_envMap, uv).rgb;

    // --- Tonemapping y balance ---
    float exposure = 0.9;             // menor exposición = menos quemado
    vec3 mapped = vec3(1.0) - exp(-hdr * exposure);

    // ajuste de contraste y brillo
    mapped = pow(mapped, vec3(1.0 / 2.2)); // gamma
    mapped = clamp(mapped * 1.1, 0.0, 1.0); // leve boost general

    FragColor = vec4(mapped, 1.0);
}
