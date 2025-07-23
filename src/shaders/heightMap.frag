#version 430

layout(location = 0) out vec4 fragmentColor;
layout(binding = 0) uniform sampler2D heightMap;
layout(binding = 1) uniform sampler2D platesTexture;
layout(binding = 2) uniform sampler2D waterTexture;
layout(binding = 3) uniform sampler2D arrowTexture;
layout(binding = 4) uniform sampler2D directionTexture;

uniform bool showBorders;
uniform bool showWater;
uniform bool showDirectionArrows;

in vec2 texCoord;
in vec4 color;

float sobelEdge(vec2 uv, sampler2D tex, vec2 texel) {
    float tl = texture(tex, uv + texel * vec2(-1, -1)).r;
    float t  = texture(tex, uv + texel * vec2( 0, -1)).r;
    float tr = texture(tex, uv + texel * vec2( 1, -1)).r;
    float l  = texture(tex, uv + texel * vec2(-1,  0)).r;
    float r  = texture(tex, uv + texel * vec2( 1,  0)).r;
    float bl = texture(tex, uv + texel * vec2(-1,  1)).r;
    float b  = texture(tex, uv + texel * vec2( 0,  1)).r;
    float br = texture(tex, uv + texel * vec2( 1,  1)).r;

    float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
    float gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;

    return sqrt(gx * gx + gy * gy);
}

void renderBorder() {
    vec2 texelSize = 1.0 / vec2(textureSize(platesTexture, 0));

    ivec2 texSize = textureSize(platesTexture, 0);
    ivec2 texelCoord = ivec2(texCoord * texSize);

    float center = texelFetch(platesTexture, texelCoord, 0).r;

    int leftWrapped = (texelCoord.x - 1 + texSize.x) % texSize.x;
    int rightWrapped = (texelCoord.x + 1) % texSize.x;
    int topWrapped = (texelCoord.y - 1 + texSize.y) % texSize.y;
    int botWrapped = (texelCoord.y + 1) % texSize.y;

    float tl = float(texelFetch(platesTexture, ivec2(leftWrapped, topWrapped), 0).r != center);
    float  t = float(texelFetch(platesTexture, ivec2(texelCoord.x, topWrapped), 0).r != center);
    float tr = float(texelFetch(platesTexture, ivec2(rightWrapped, topWrapped), 0).r != center);
    float  l = float(texelFetch(platesTexture, ivec2(leftWrapped,  texelCoord.y), 0).r != center);
    float  r = float(texelFetch(platesTexture, ivec2(rightWrapped,  texelCoord.y), 0).r != center);
    float bl = float(texelFetch(platesTexture, ivec2(leftWrapped,  botWrapped), 0).r != center);
    float  b = float(texelFetch(platesTexture, ivec2(texelCoord.x,  botWrapped), 0).r != center);
    float br = float(texelFetch(platesTexture, ivec2(rightWrapped,  botWrapped), 0).r != center);

    // Apply Sobel operator to binary edges
    float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
    float gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;

    float sobel = sqrt(gx * gx + gy * gy);  // Range: 0 to 8

    // Optional smoothstep for AA/falloff
    float edgeAlpha = smoothstep(0.5, 3.0, sobel);

    vec4 borderColor = vec4(1.0, 0.0, 0.0, 1.0); // red border

    fragmentColor = mix(fragmentColor, borderColor, edgeAlpha);
}

void drawDirectionArrows()
{
    vec2 arrowUV = mod(texCoord * textureSize(heightMap, 0) / 16, 1.0);

    ivec2 texSize = textureSize(directionTexture, 0);
    ivec2 texelCoord = ivec2(texCoord * texSize);
    vec2 dir = texelFetch(directionTexture, texelCoord, 0).xy;

    float angle = atan(dir.y, dir.x);

    // Rotate centered UV
    float c = cos(-angle);
    float s = sin(-angle);
    mat2 rot = mat2(c, -s, s, c);

    vec2 centeredUV = arrowUV - 0.5;
    vec2 rotatedUV = rot * centeredUV + 0.5;

    vec4 arrowColor = texture(arrowTexture, rotatedUV);
    vec3 mixedColor = mix(fragmentColor.rgb, arrowColor.rgb, arrowColor.a);
    fragmentColor = vec4(mixedColor, 1);
//    fragmentColor = vec4(dir.x, dir.y, 1, 1);
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

    if (showDirectionArrows)
    {
        drawDirectionArrows();
    }

    if (showBorders)
    {
        renderBorder();
    }
}
