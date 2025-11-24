#version 330 core
in vec2 vPosNDC;
out vec4 FragColor;

uniform sampler2D u_envMap;
uniform mat4 u_invProj;
uniform mat3 u_invViewRot;

void main()
{
    vec4 ndc = vec4(vPosNDC, 1.0, 1.0);
    vec4 viewDirH = u_invProj * ndc;
    vec3 viewDir  = normalize(viewDirH.xyz / viewDirH.w);
    vec3 dir      = normalize(u_invViewRot * viewDir);

    float PI = 3.14159265;
    vec2 uv = vec2(
        atan(dir.z, dir.x) / (2.0 * PI) + 0.5,
        asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5
    );

    vec3 hdr = texture(u_envMap, uv).rgb;

    // Como ya normalizamos el HDR en CPU, solo gamma:
    vec3 mapped = pow(hdr, vec3(1.0 / 2.2));
    FragColor = vec4(mapped, 1.0);
}
