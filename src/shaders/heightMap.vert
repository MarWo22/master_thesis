#version 430

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

layout(binding = 0) uniform sampler2D heightMap;

uniform mat4 vpMat;

out vec4 color;

void main() {
    const float height = texture(heightMap, texCoord).r;
    color = vec4(vec3(height), 1);
    gl_Position = vpMat * vec4(position.x, height * 200,  position.z, 1.0);
}
