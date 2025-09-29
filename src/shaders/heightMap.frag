#version 430

layout(location = 0) out vec4 fragmentColor;
layout(binding = 0) uniform sampler2D heightMap;
layout(binding = 1) uniform sampler2D platesTexture;
layout(binding = 2) uniform sampler2D waterTexture;
layout(binding = 3) uniform sampler2D arrowTexture;
layout(binding = 4) uniform sampler2D directionTexture;
layout(binding = 5) uniform usampler2D collisionTexture;

uniform int borderRenderType;
uniform bool showWater;
uniform bool showDirectionArrows;
uniform int shadingType;
uniform float continentalCrustThreshold;

in vec2 texCoord;
in vec4 color;



float sobelEdge(vec2 uv, sampler2D tex, vec2 texel) {
    float tl = texture(tex, uv + texel * vec2(-1, -1)).r;
    float t  = texture(tex, uv + texel * vec2(0, -1)).r;
    float tr = texture(tex, uv + texel * vec2(1, -1)).r;
    float l  = texture(tex, uv + texel * vec2(-1, 0)).r;
    float r  = texture(tex, uv + texel * vec2(1, 0)).r;
    float bl = texture(tex, uv + texel * vec2(-1, 1)).r;
    float b  = texture(tex, uv + texel * vec2(0, 1)).r;
    float br = texture(tex, uv + texel * vec2(1, 1)).r;

    float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
    float gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;

    return sqrt(gx * gx + gy * gy);
}

void renderSmoothBorder() {
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
    float  l = float(texelFetch(platesTexture, ivec2(leftWrapped, texelCoord.y), 0).r != center);
    float  r = float(texelFetch(platesTexture, ivec2(rightWrapped, texelCoord.y), 0).r != center);
    float bl = float(texelFetch(platesTexture, ivec2(leftWrapped, botWrapped), 0).r != center);
    float  b = float(texelFetch(platesTexture, ivec2(texelCoord.x, botWrapped), 0).r != center);
    float br = float(texelFetch(platesTexture, ivec2(rightWrapped, botWrapped), 0).r != center);

    // Apply Sobel operator to binary edges
    float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
    float gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;

    float sobel = sqrt(gx * gx + gy * gy);// Range: 0 to 8

    // Optional smoothstep for AA/falloff
    float edgeAlpha = smoothstep(0.5, 3.0, sobel);

    vec4 borderColor = vec4(1.0, 0.0, 0.0, 1.0);// red border

    fragmentColor = mix(fragmentColor, borderColor, edgeAlpha);
}

void renderRawBorder() {
    ivec2 texSize = textureSize(platesTexture, 0);
    ivec2 texelCoord = ivec2(texCoord * texSize);

    // Manual wrapping with modulo
    ivec2 leftCoord = ivec2((texelCoord.x - 1 + texSize.x) % texSize.x, texelCoord.y);
    ivec2 rightCoord = ivec2((texelCoord.x + 1) % texSize.x, texelCoord.y);
    ivec2 topCoord = ivec2(texelCoord.x, (texelCoord.y + 1) % texSize.y);
    ivec2 botCoord = ivec2(texelCoord.x, (texelCoord.y - 1 + texSize.y) % texSize.y);

    float region = texelFetch(platesTexture, texelCoord, 0).r;
    float leftRegion = texelFetch(platesTexture, leftCoord, 0).r;
    float rightRegion = texelFetch(platesTexture, rightCoord, 0).r;
    float topRegion = texelFetch(platesTexture, topCoord, 0).r;
    float botRegion = texelFetch(platesTexture, botCoord, 0).r;

    if (region != leftRegion || region != rightRegion || region != botRegion || region != topRegion)
    fragmentColor = vec4(1, 0, 0, 1);
}

void renderCollisionBorder() {
    ivec2 texelCoord = ivec2(texCoord * textureSize(collisionTexture, 0));
    uint value = texelFetch(collisionTexture, texelCoord, 0).r;

    if (value == 0)
    fragmentColor = vec4(1, 1, 1, 1);
    else if (((value >> 8) & 0xFFu) != 0u)
    fragmentColor = vec4(0, 0, 0, 1);
}

void drawDirectionArrows()
{
    vec2 arrowUV = mod(texCoord * textureSize(heightMap, 0) / 16, 1.0);
    arrowUV.y = 1.0 - arrowUV.y;
    ivec2 texSize = textureSize(directionTexture, 0);
    ivec2 texelCoord = ivec2(texCoord * texSize);
    vec2 dir = texelFetch(directionTexture, texelCoord, 0).xy;

    if (dir.x == 0 && dir.y == 0)
    return;

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


vec3 normalShading(float height)
{
    if (height < 4000) {
        return mix(vec3(0.612, 0.6, 0.514), vec3(0.878, 0.859, 0.722), height / 4000);
    } else if (height < 8000) {
        return mix(vec3(0.878, 0.859, 0.722), vec3(0.949, 0.933, 0.82), (height - 4000) / 4000);
    } else if (height < 12000) {
        return mix(vec3(0.102, 0.541, 0.141), vec3(0.216, 0.478, 0.239), (height - 8000) / 4000);
    }

    return mix(vec3(0.451, 0.451, 0.451), vec3(0.71, 0.71, 0.71), (height - 12000) / 20000);
}

vec3 crustTypeShading(float height)
{
    if (height < continentalCrustThreshold)
    {
        // Ocean shading: deep blue to shallow turquoise
        float t = height / continentalCrustThreshold;
        vec3 deep = vec3(0.0, 0.0, 0.3);// Deep ocean
        vec3 shallow = vec3(0.0, 0.7, 1.0);// Near shore
        return mix(deep, shallow, t);
    }

    // Land shading
    float landHeight = height - continentalCrustThreshold;

    if (landHeight < 200.0) {
        // Coastal soil/dirt (0–200 m)
        return mix(vec3(0.4, 0.3, 0.2), vec3(0.6, 0.5, 0.3), landHeight / 200.0);

    } else if (landHeight < 1000.0) {
        // Transition to grass (200–1000 m)
        return mix(vec3(0.6, 0.5, 0.3), vec3(0.1, 0.6, 0.1), (landHeight - 200.0) / 800.0);

    } else if (landHeight < 2500.0) {
        // Vivid green grass to alpine green (1000–2500 m)
        return mix(vec3(0.1, 0.6, 0.1), vec3(0.2, 0.4, 0.2), (landHeight - 1000.0) / 1500.0);

    } else if (landHeight < 4000.0) {
        // Grass to rocky gray (2500–4000 m)
        return mix(vec3(0.2, 0.4, 0.2), vec3(0.5, 0.5, 0.5), (landHeight - 2500.0) / 1500.0);
    }

    // Very high elevations – rocky → snowy (4000–10000 m)
    return mix(vec3(0.5, 0.5, 0.5), vec3(0.9, 0.9, 0.95), clamp((landHeight - 4000.0) / 6000.0, 0.0, 1.0));
}

// Function to convert HSV to RGB
vec3 hsvToRgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0 - abs(mod(h * 6.0, 2.0) - 1.0));
    float m = v - c;

    float r, g, b;

    if (h >= 0.0 && h < 1.0 / 6.0) {
        r = c;
        g = x;
        b = 0.0;
    } else if (h >= 1.0 / 6.0 && h < 2.0 / 6.0) {
        r = x;
        g = c;
        b = 0.0;
    } else if (h >= 2.0 / 6.0 && h < 3.0 / 6.0) {
        r = 0.0;
        g = c;
        b = x;
    } else if (h >= 3.0 / 6.0 && h < 4.0 / 6.0) {
        r = 0.0;
        g = x;
        b = c;
    } else if (h >= 4.0 / 6.0 && h < 5.0 / 6.0) {
        r = x;
        g = 0.0;
        b = c;
    } else {
        r = c;
        g = 0.0;
        b = x;
    }

    return vec3(r + m, g + m, b + m);
}


vec3 getColorFromID(int id) {
    // Map the ID to a hue value between 0 and 1 (to cover the entire hue spectrum)
    float hue = 0.125 * (id % 8) + float(id / 8) / 256.0;


    // Set saturation and value to maximum for bright colors
    float saturation = 0.5 + 0.5 * sin(float(id) * 0.1);// Oscillating between 0.0 and 1.0
    float value = 0.5 + 0.5 * cos(float(id) * 0.1);// Oscillating between 0.0 and 1.0

    // Convert HSV to RGB
    return hsvToRgb(hue, saturation, value);
}

vec3 plateIdShading()
{
    ivec2 texelCoord = ivec2(texCoord * textureSize(platesTexture, 0));
    float normalizedID = texelFetch(platesTexture, texelCoord, 0).r;

    int id = int(normalizedID * 255);
    return getColorFromID(id);
}

void main() {
    float height = color.r;

    vec3 colorNew;


    switch (shadingType)
    {
        case 0:
        colorNew = normalShading(height);
        break;
        case 1:
        colorNew = crustTypeShading(height);
        break;
        case 2:
        colorNew = plateIdShading();
        break;
    }

    if (showWater){
        float hydration = texture(waterTexture, texCoord).r;
        if (hydration > 20)
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

    switch (borderRenderType)
    {
        case 0:
        // No border is rendered
        break;
        case 1:
        renderRawBorder();
        break;
        case 2:
        renderSmoothBorder();
        break;
        case 3:
        renderCollisionBorder();
        break;
    }
}
