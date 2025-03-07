#include <iostream>
#include "random_texture.h"
#include "texture_save.h"
#include "flux_erosion.h"
#include <string>

int main()
{
    auto input = generateRandomTexture(1024, 1024);
    
    FluxVelocityErosion fluxVelocityErosion(1024, 1024);
    fluxVelocityErosion.simulate(input, 100);

    float* eroded = new float[1024 * 1024];

    fluxVelocityErosion.getMaterial(eroded);

    auto texture16Bit = convertFloatTextureTo16Bit(eroded, 1024, 1024);

    std::time_t now = std::time(nullptr);
    auto filename = std::string("output_") + std::to_string(now) + ".png";

    Texture::save16BitGreyscalePng(filename.c_str(), texture16Bit, 1024, 1024);

    delete eroded;
    delete texture16Bit;
    return 0;
}

