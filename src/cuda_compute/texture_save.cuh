#ifndef TEXTURE_SAVE_H
#define TEXTURE_SAVE_H
#include <iostream>

#include "cuda_texture.cuh"

#include <vector>
#include <cuda_runtime.h>
#include <png.h>

// Forward declare helper function(s) in a detail namespace
namespace texture_save_detail {
    template<typename T>
    void saveGrayscaleTextureToDisk(const char *fileName, T *imageData, int width, int height)
    {
        FILE *fp;
        if (fopen_s(&fp, fileName, "wb") != 0)
        {
            std::cerr << "Error opening file\n";
            return;
        }

        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        png_infop info_ptr = png_create_info_struct(png_ptr);

        if (!png_ptr || !info_ptr || setjmp(png_jmpbuf(png_ptr)))
        {
            std::cerr << "PNG setup failed\n";
            if (png_ptr && info_ptr) png_destroy_write_struct(&png_ptr, &info_ptr);
            fclose(fp);
            return;
        }

        png_init_io(png_ptr, fp);
        png_set_IHDR(png_ptr, info_ptr, width, height, sizeof(T) * 8,
                     PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png_ptr, info_ptr);

        std::vector<png_bytep> row_pointers(height);
        for (int y = 0; y < height; y++)
            row_pointers[y] = reinterpret_cast<png_bytep>(&imageData[y * width]);

        png_write_image(png_ptr, row_pointers.data());
        png_write_end(png_ptr, nullptr);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
    }
}

// Define the template function here (can call detail helpers)
template<typename T>
void saveCudaTextureToDiskGrayscale(const char* fileName, const CudaTextureHost<T> &texture)
{
    int width = texture.width();
    int height = texture.height();

    std::vector<T> imageData(width * height);

    if (const cudaError_t err = cudaMemcpy(imageData.data(), texture.getPointer(), sizeof(T) * width * height,
                                           cudaMemcpyDeviceToHost); err != cudaSuccess)
    {
        std::cerr << "Error copy texture during disk save: " << cudaGetErrorString(err) << std::endl;
    }

    texture_save_detail::saveGrayscaleTextureToDisk(fileName, imageData.data(), width, height);
}


void saveGrayscale8BitCudaTextureToDiskAsRgb(const char *fileName, const CudaTextureHost<uint8_t> &texture);




#endif //TEXTURE_SAVE_H
