#version 430

layout(location = 0) out vec4 fragmentColor;
layout(binding = 0) uniform sampler2D heightMap;
layout(binding = 1) uniform sampler2D platesTexture;

uniform bool showBorders;

in vec2 texCoord;
in vec4 color;

void renderBorder() {
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

}


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
    if (showBorders)
        renderBorder();
}
