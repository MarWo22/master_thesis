#include <algorithm>
#include <iostream>
#include "random_texture.h"
#include "texture_save.h"
#include "flux_erosion.h"
#include <string>


uint8_t *convertTextureTemp(const float *texture, int width, int height)
{
    auto *texture8bit = new uint8_t[1024*1024];
    for (int i = 0; i < 1024*1024; ++i)
        texture8bit[i] = static_cast<uint8_t>(std::clamp(texture[i] * 255.0f, 0.0f, 255.0f));
    return texture8bit;
}


int main()
{
    auto input = generateRandomTexture(1024, 1024);
    
    FluxVelocityErosion fluxVelocityErosion(1024, 1024);
    fluxVelocityErosion.simulate(input, 1000);

    float* output = new float[1024 * 1024];

    fluxVelocityErosion.getHydration(output);

    std::time_t now = std::time(nullptr);
    auto filename = std::string("output_") + std::to_string(now) + "_hydration.png";

    auto texture8Bit = convertTextureTemp(output, 1024, 1024);
    Texture::save8BitGreyscalePng(filename.c_str(), texture8Bit, 1024, 1024);
    delete texture8Bit;

    fluxVelocityErosion.getMaterial(output);

    filename = std::string("output_") + std::to_string(now) + "_material.png";

    texture8Bit = convertTextureTemp(output, 1024, 1024);
    Texture::save8BitGreyscalePng(filename.c_str(), texture8Bit, 1024, 1024);
    delete texture8Bit;

    fluxVelocityErosion.getSediment(output);

    filename = std::string("output_") + std::to_string(now) + "_sediment.png";

    texture8Bit = convertTextureTemp(output, 1024, 1024);
    Texture::save8BitGreyscalePng(filename.c_str(), texture8Bit, 1024, 1024);
    delete texture8Bit;

    delete output;
    return 0;
}

