#include <iostream>
#include "random_texture.h"
#include "texture_save.h"
#include "flux_erosion.h"
#include <string>

int main()
{
    int size = 256;

    auto input = generateRandomTexture(size, size);
    
    FluxVelocityErosion fluxVelocityErosion(size, size);
    fluxVelocityErosion.simulate(input, 10);

    float* output = new float[size * size];

    fluxVelocityErosion.getHydration(output);

    auto texture16Bit = convertFloatTextureTo16Bit(output, size, size);

    std::time_t now = std::time(nullptr);
    auto filename = std::string("output_") + std::to_string(now) + "_hydration.png";

    Texture::save16BitGreyscalePng(filename.c_str(), texture16Bit, size, size);

    fluxVelocityErosion.getMaterial(output);

    texture16Bit = convertFloatTextureTo16Bit(output, size, size);

    filename = std::string("output_") + std::to_string(now) + "_material.png";

    Texture::save16BitGreyscalePng(filename.c_str(), texture16Bit, size, size);

    fluxVelocityErosion.getSediment(output);

    texture16Bit = convertFloatTextureTo16Bit(output, size, size);

    filename = std::string("output_") + std::to_string(now) + "_sediment.png";

    Texture::save16BitGreyscalePng(filename.c_str(), texture16Bit, size, size);

    delete output;
    delete texture16Bit;
    return 0;
}

