#version 430

layout(location = 0) out vec4 fragmentColor;
layout(binding = 0) uniform sampler2D heightMap;

in vec4 color;

void main() {
    fragmentColor = color;
}
