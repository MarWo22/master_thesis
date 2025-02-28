#ifndef RANDOM_TEXTURE_H
#define RANDOM_TEXTURE_H

float* generateRandomTexture(int height, int width);
uint16_t* convertFloatTextureTo16Bit(float *texture, int height, int width);

#endif //RANDOM_TEXTURE_H
