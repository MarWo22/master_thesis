#include <algorithm>
#include <iostream>
#include "texture_save.h"
#include "erosion.h"
#include "particle_erosion.h"

int main()
{
    // auto texture = generateRandomTexture(1024, 1024, 1337);
    float *texture = generateRandomTexture(1024, 1024, 1337);
    uint8_t *texture8bit = new uint8_t[1024*1024];
    for (int i = 0; i < 1024*1024; ++i)
        texture8bit[i] = static_cast<uint8_t>(std::clamp(texture[i] * 255.0f, 0.0f, 255.0f));

    Texture::save8BitGreyscalePng("before_erosion.png", texture8bit, 1024, 1024);

    ParticleErosion particleErosion(1024, 1024);
    particleErosion.loadHeightmap(texture);
    particleErosion.simulateDroplets(1250000, 1337);
    particleErosion.extractHeightmap(texture);
    uint8_t *texture8bitErosion = new uint8_t[1024*1024];

    for (int i = 0; i < 1024*1024; ++i)
        texture8bitErosion[i] = static_cast<uint8_t>(std::clamp(texture[i] * 255.0f, 0.0f, 255.0f));

    int count = 0;
    for (int i = 0; i != 1024*1024; ++i)
        if (texture8bit[i] != texture8bitErosion[i])
            count++;

    std::cout << "Mismatched percentage: " << count / (1024.0*1024.0) * 100 << "\n";

    Texture::save8BitGreyscalePng("erosion.png", texture8bitErosion, 1024, 1024);

    delete texture;
    delete texture8bit;
    delete texture8bitErosion;
}

