#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D u_screenTex;
uniform int  u_enableFisheye;
uniform float u_strength; // 0..1

void main()
{
    vec2 uv = vUV;

    if (u_enableFisheye == 1 && u_strength > 0.001) {
        // Centro de pantalla
        vec2 center = vec2(0.5, 0.5);
        vec2 coord = uv - center;

        float r = length(coord);
        if (r < 0.75) {
            // Fisheye
            float theta = atan(coord.y, coord.x);
            float radius = pow(r, 1.0 + u_strength * 1.5);
            coord = radius * vec2(cos(theta), sin(theta));
            uv = center + coord;
        }
    }

    // Aberración cromática ligera
    float chroma = u_strength * 0.015;

    vec3 col;
    col.r = texture(u_screenTex, uv + vec2( chroma,  0.0)).r;
    col.g = texture(u_screenTex, uv).g;
    col.b = texture(u_screenTex, uv + vec2(-chroma, 0.0)).b;

    FragColor = vec4(col, 1.0);
}
