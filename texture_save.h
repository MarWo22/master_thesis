#ifndef TEXTURE_SAVE_H
#define TEXTURE_SAVE_H
#include <iostream>

namespace Texture
{
    void save16BitGreyscalePng(const char *fileName, float *image_data, int width, int height);

    void save16BitGreyscalePng(const char *fileName, uint16_t *image_data, int width, int height);

    void save8BitGreyscalePng(const char *fileName, uint8_t *image_data, int width, int height);
}



#endif //TEXTURE_SAVE_H
