#version 330 core
in vec3 vTexCoord;
out vec4 FragColor;
uniform sampler2D u_envMap;

void main() {
    vec3 d = normalize(vTexCoord);
    vec2 uv = vec2(
        atan(d.z, d.x) / (2.0 * 3.14159265) + 0.5,
        asin(d.y) / 3.14159265 + 0.5
    );

    vec3 hdr = texture(u_envMap, uv).rgb;

    // tonemap + gamma
    float exposure = 1.6;
    vec3 mapped = vec3(1.0) - exp(-hdr * exposure);
    mapped = pow(mapped, vec3(1.0/2.2));
    FragColor = vec4(mapped, 1.0);
}
