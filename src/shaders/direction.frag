#version 430

layout(location = 0) out vec4 fragmentColor;
layout(binding = 1) uniform sampler2D directionTexture;

in vec2 texCoord;

void main() {
    ivec2 texelCoord = ivec2(texCoord * textureSize(directionTexture, 0));
    vec2 direction = texelFetch(directionTexture, texelCoord, 0).rg;
    vec2 normalized_direction = (direction + 1) * 0.5f;
    fragmentColor = vec4(direction.x, 0, direction.y, 1);
}
