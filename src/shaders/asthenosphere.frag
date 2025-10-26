#version 430

layout(location = 0) out vec4 fragmentColor;
layout(binding = 1) uniform sampler2D asthenosphereTexture;

in vec2 texCoord;

void main() {
    ivec2 texelCoord = ivec2(texCoord * textureSize(asthenosphereTexture, 0));
    vec2 velocity = texelFetch(asthenosphereTexture, texelCoord, 0).rg;

    // Calculate magnitude for debugging
    float magnitude = length(velocity);

    // Scale up significantly for visualization
    float visualizationScale = 1000.0;
    float scaledMagnitude = magnitude * visualizationScale;

    // Show as grayscale magnitude for debugging
    // fragmentColor = vec4(vec3(scaledMagnitude), 1.0);

    // Scale the vector components
    vec2 scaledVelocity = velocity * visualizationScale;

    // Clamp to prevent oversaturation
    scaledVelocity = clamp(scaledVelocity, -1.0, 1.0);

    // Map from [-1, 1] to [0, 1] for color display
    vec2 colorVelocity = (scaledVelocity + 1.0) * 0.5;

    fragmentColor = vec4(colorVelocity.x, 0, colorVelocity.y, 1);
}
