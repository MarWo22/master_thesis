#version 430

layout(location = 0) out vec4 fragmentColor;
layout(binding = 0) uniform sampler2D heightMap;

in vec4 color;

void main() {
    
    float height = color.r;

    vec3 colorNew = vec3(color.r, color.g, color.b);

    if (height < 0.1) {
        colorNew = mix(vec3(0.0, 0.0, 0.3), vec3(0.0, 0.5, 1.0), height / 0.1);
    } else if (height < 0.6) {
        colorNew = mix(vec3(0.0, 0.5, 0.0), vec3(0.4, 0.3, 0.2), (height - 0.1) / 0.3);
    } else if (height < 0.85) {
        colorNew = mix(vec3(0.4, 0.3, 0.2), vec3(0.5, 0.5, 0.5), (height - 0.6) / 0.25);
    } else {
        colorNew = mix(vec3(0.5, 0.5, 0.5), vec3(1.0, 1.0, 1.0), (height - 0.85) / 0.15);
    }

    fragmentColor = vec4(colorNew, 1.0);    
}
