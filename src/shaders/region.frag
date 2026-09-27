#version 430

layout(location = 0) out vec4 fragmentColor;
layout(binding = 1) uniform sampler2D regionTexture;

in vec2 texCoord;

void main() {
    ivec2 texelCoord = ivec2(texCoord * textureSize(regionTexture, 0));
    float region = texelFetch(regionTexture, texelCoord, 0).r;
    if (region == 0)
        fragmentColor = vec4(0, 0, 0 ,1);
    else if (region <= 0.25)
        fragmentColor = vec4(region * 4, 0, 0,1);
    else if (region <= 0.5)
        fragmentColor = vec4(0, (region - 0.25) * 4, 0,1);
    else if (region <= 0.75)
        fragmentColor = vec4(0, 0, (region - 0.5) * 4,1);
    else
        fragmentColor = vec4(region, region, region ,1);
}
