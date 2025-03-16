#include <algorithm>
#include <iostream>
#include "FastNoiseLite.h"
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

float* generateRandomTexture(int height, int width, int seed)
{
    // Allocate array on device memory
    int size = height * width;
    // Allocate on host memory
    auto hTexture = new float[size];

    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    noise.SetSeed(seed);
    noise.SetFrequency(0.001f);
    noise.SetFractalType(FastNoiseLite::FractalType_FBm);
    noise.SetFractalOctaves(5);

    int index = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            hTexture[index++] = (noise.GetNoise(static_cast<float>(x), static_cast<float>(y)) + 1) / 2;
        }
    }
    return hTexture;
}

int main()
{
    int size = 1024;

    auto input = generateRandomTexture(size, size, 1023123132);
    
    std::time_t now = std::time(nullptr);
    quickExport(input, "start_material", 0, now);

    float* output = new float[size * size];



    FluxVelocityErosion fluxVelocityErosion(size, size);

    fluxVelocityErosion.mPipeCrossSectionConstant = .5;
    fluxVelocityErosion.mPipeLengthConstant = .5;
    fluxVelocityErosion.mCapacityConstant = 4;
    fluxVelocityErosion.mDissolvingConstant = 0.01;
    fluxVelocityErosion.mEvaporationConstant = 0.9;
    fluxVelocityErosion.mGravityConstant = 9.81;

    fluxVelocityErosion.start(input, 1);
    fluxVelocityErosion.getMaterial(output);
    quickExport(output, "material", 0, now);

    for (int i = 0; i < 5; i++)
    {
        fluxVelocityErosion.resume(100);
        fluxVelocityErosion.getMaterial(output);
        quickExport(output, "material", i + 1, now);

        fluxVelocityErosion.getHydration(output);
        quickExport(output, "hydration", i + 1, now);

        fluxVelocityErosion.getSediment(output);
        quickExport(output, "sediment", i + 1, now);
    }

    

    


    

    delete output;
    return 0;
}



