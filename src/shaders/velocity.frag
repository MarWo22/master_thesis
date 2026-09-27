#version 430

layout(location = 0) out vec4 fragmentColor;
layout(binding = 1) uniform sampler2D velocityTexture;

in vec2 texCoord;

void main() {
    ivec2 texelCoord = ivec2(texCoord * textureSize(velocityTexture, 0));
    float velocity = texelFetch(velocityTexture, texelCoord, 0).r;

    float clamped_velocity = min(velocity, 2.5); // Change as needed

    fragmentColor = vec4(vec3(clamped_velocity / 2.5), 1);
}
