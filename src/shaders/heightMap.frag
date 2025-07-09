#version 430

layout(location = 0) out vec4 fragmentColor;
layout(binding = 0) uniform sampler2D heightMap;
layout(binding = 1) uniform sampler2D platesTexture;
layout(binding = 2) uniform sampler2D waterTexture;

uniform bool showBorders;
uniform bool showWater;

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

    if (height < 100) {
        colorNew = mix(vec3(0.612, 0.6, 0.514), vec3(0.878, 0.859, 0.722), height / 100);
    } else if (height < 200) {
        colorNew = mix(vec3(0.878, 0.859, 0.722), vec3(0.949, 0.933, 0.82), (height - 100) / 100);
    } else if (height < 300) {
        colorNew = mix(vec3(0.102, 0.541, 0.141), vec3(0.216, 0.478, 0.239), (height - 200) / 100);
    } else {
        colorNew = mix(vec3(0.451, 0.451, 0.451), vec3(0.71, 0.71, 0.71), (height - 300) / 500);
    }
    
    if (showWater){
        float hydration = texture(waterTexture, texCoord).r;
        if(hydration > 20)
        {
            vec4 water = mix(vec4(0.639, 0.949, 1, 0.6), vec4(0, 0.588, 0.588, 1.0), clamp(hydration, 0, 100) / 100);
            colorNew = mix(colorNew, water.rgb, water.a);
        }
        
    }
    
    
    fragmentColor = vec4(colorNew, 1.0);    
    if (showBorders)
        renderBorder();

    
    
}
