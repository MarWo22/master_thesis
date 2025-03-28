#version 430

layout(location = 0) out vec4 fragmentColor;
layout(binding = 1) uniform sampler2D platesTexture;
layout(binding = 0) uniform sampler2D heightMap;

uniform vec2 textureSize;

in vec2 texCoord;

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
    float saturation = 0.5 + 0.5 * sin(float(id) * 0.1);  // Oscillating between 0.0 and 1.0
    float value = 0.5 + 0.5 * cos(float(id) * 0.1);  // Oscillating between 0.0 and 1.0

    // Convert HSV to RGB
    return hsvToRgb(hue, saturation, value);
}

void main() {
    float normalizedID = texture(platesTexture, texCoord).r;
    int id = int(normalizedID * 255);
    fragmentColor = vec4(getColorFromID(id),1);
}
