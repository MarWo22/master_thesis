#version 430

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoordIn;

layout(binding = 0) uniform sampler2D heightMap;

uniform mat4 vpMat;
uniform bool displaceHeight;
uniform float heightMultiplier;

out vec4 color;
out vec2 texCoord;

void main() {
    const float height = texture(heightMap, texCoordIn).r;
    texCoord = texCoordIn;
    color = vec4(vec3(height), 1);

    if (displaceHeight)
        gl_Position = vpMat * vec4(position.x, height * heightMultiplier, position.z, 1.0);
    else
        gl_Position = vpMat * vec4(position.x, 0, position.z, 1.0);

}
