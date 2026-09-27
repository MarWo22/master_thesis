#version 430

layout(location = 0) out vec4 fragmentColor;
layout(binding = 1) uniform sampler2D platesTexture;

in vec2 texCoord;

void main() {
    ivec2 texSize = textureSize(platesTexture, 0);
    ivec2 texelCoord = ivec2(texCoord * texSize);

    // Manual wrapping with modulo
    ivec2 leftCoord  = ivec2((texelCoord.x - 1 + texSize.x) % texSize.x, texelCoord.y);
    ivec2 rightCoord = ivec2((texelCoord.x + 1) % texSize.x, texelCoord.y);
    ivec2 topCoord   = ivec2(texelCoord.x, (texelCoord.y + 1) % texSize.y);
    ivec2 botCoord   = ivec2(texelCoord.x, (texelCoord.y - 1 + texSize.y) % texSize.y);

    float region      = texelFetch(platesTexture, texelCoord, 0).r;
    float leftRegion  = texelFetch(platesTexture, leftCoord, 0).r;
    float rightRegion = texelFetch(platesTexture, rightCoord, 0).r;
    float topRegion   = texelFetch(platesTexture, topCoord, 0).r;
    float botRegion   = texelFetch(platesTexture, botCoord, 0).r;

    if (region != leftRegion || region != rightRegion || region != botRegion || region != topRegion)
        fragmentColor = vec4(1, 0, 0, 1);
    else
        fragmentColor = vec4(0, 0, 0, 1);
}
