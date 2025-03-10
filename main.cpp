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


void quickExport(float* data, std::string tag, int batch, std::time_t now) {
    auto filename = std::string("output_") + std::to_string(now) + "_" + tag + "_" + std::to_string(batch) + ".png";

    auto texture8Bit = convertTextureTemp(data, 1024, 1024);
    Texture::save8BitGreyscalePng(filename.c_str(), texture8Bit, 1024, 1024);
    delete texture8Bit;
}

int main()
{
    auto input = generateRandomTexture(1024, 1024);
    float* output = new float[1024 * 1024];
    std::time_t now = std::time(nullptr);

    FluxVelocityErosion fluxVelocityErosion(1024, 1024);

    fluxVelocityErosion.start(input, 1000);
    fluxVelocityErosion.getHydration(output);
    quickExport(output, "hydration", 0, now);

    fluxVelocityErosion.resume(1000);
    fluxVelocityErosion.getHydration(output);
    quickExport(output, "hydration", 1, now);

    fluxVelocityErosion.resume(1000);
    fluxVelocityErosion.getHydration(output);
    quickExport(output, "hydration", 2, now);


    fluxVelocityErosion.getHydration(output);

    

    delete output;
    return 0;
}



