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

    float* output = new float[1024 * 1024];

    fluxVelocityErosion.getHydration(output);

    auto texture16Bit = convertFloatTextureTo16Bit(output, 1024, 1024);

    std::time_t now = std::time(nullptr);
    auto filename = std::string("output_") + std::to_string(now) + "_hydration.png";

    Texture::save16BitGreyscalePng(filename.c_str(), texture16Bit, 1024, 1024);

    fluxVelocityErosion.getMaterial(output);

    texture16Bit = convertFloatTextureTo16Bit(output, 1024, 1024);

    filename = std::string("output_") + std::to_string(now) + "_material.png";

    Texture::save16BitGreyscalePng(filename.c_str(), texture16Bit, 1024, 1024);

    fluxVelocityErosion.getSediment(output);

    texture16Bit = convertFloatTextureTo16Bit(output, 1024, 1024);

    filename = std::string("output_") + std::to_string(now) + "_sediment.png";

    Texture::save16BitGreyscalePng(filename.c_str(), texture16Bit, 1024, 1024);

    delete output;
    delete texture16Bit;
    return 0;
}

