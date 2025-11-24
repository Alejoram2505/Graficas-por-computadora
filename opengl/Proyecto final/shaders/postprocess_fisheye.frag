#version 330 core

in vec2 vUV;              // <--- MISMO nombre que en el vertex
out vec4 FragColor;

uniform sampler2D u_screenTex;

// toggles
uniform int  u_enableFisheye;
uniform int  u_enableChromAb;
uniform float u_strength;   // intensidad general (I / O en Python)

void main()
{
    vec2 uv = vUV;

    // Centro [-1,1]
    vec2 centered = uv * 2.0 - 1.0;
    float r = length(centered);
    vec2 dir = (r > 0.0001) ? centered / r : vec2(0.0);

    // ---- FISHEYE ----
    vec2 uvDist = uv;
    if (u_enableFisheye == 1)
    {
        float k = u_strength;            // 0.0 -> sin distorsión, >0 más fuerte
        float r2 = pow(r, 1.0 + k * 2.0);
        uvDist = dir * r2 * 0.5 + 0.5;
    }

    uvDist = clamp(uvDist, 0.001, 0.999);

    // ---- ABERRACIÓN CROMÁTICA ----
    vec3 color;
    if (u_enableChromAb == 1)
    {
        // desplazamiento depende un poco de la intensidad
        float ca = 0.002 + 0.008 * u_strength;
        vec2 offset = dir * ca;

        float rC = texture(u_screenTex, uvDist + offset).r;
        float gC = texture(u_screenTex, uvDist).g;
        float bC = texture(u_screenTex, uvDist - offset).b;
        color = vec3(rC, gC, bC);
    }
    else
    {
        color = texture(u_screenTex, uvDist).rgb;
    }

    FragColor = vec4(color, 1.0);
}
