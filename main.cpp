#include <iostream>
#include "random_texture.h"
#include "texture_save.h"
#include "erosion.h"

int main()
{

    auto randomTexture = run_erosion(1024, 1024);

    auto texture16Bit = convertFloatTextureTo16Bit(randomTexture, 1024, 1024);

    Texture::save16BitGreyscalePng("random_texture.png", texture16Bit, 1024, 1024);

    delete randomTexture;
    delete texture16Bit;
    return 0;
}

