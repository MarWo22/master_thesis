#include <iostream>
#include "random_texture.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>


void save16BitGreyscalePng(const char *filename, uint16_t *image_data, int width, int height)
{
    int stride = width * sizeof(uint16_t);

    if (stbi_write_png(filename, width, height, 1, image_data, stride))
        std::cout << "Successfully saved image " << filename << std::endl;
    else
        std::cout << "Failed to save image " << filename << std::endl;

}


int main()
{

    auto randomTexture = generateRandomTexture(1024, 1024);
    auto texture16Bit = convertFloatTextureTo16Bit(randomTexture, 1024, 1024);

    save16BitGreyscalePng("random_texture.png", texture16Bit, 1024, 1024);

    delete randomTexture;
    delete texture16Bit;

    return 0;
}

