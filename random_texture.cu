#include "random_texture.h"

#include <iostream>

__global__ void generateRandomTexture()
{
    unsigned int idx = blockIdx.x * blockIdx.x + threadIdx.x;
    printf("Index %d\n",idx);
}

void generateTexture()
{
    generateRandomTexture<<<5, 5>>>();
}


