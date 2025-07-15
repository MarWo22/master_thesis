#include "texture_save.cuh"

#include <png.h>
#include <vector>

#include "cuda_helper.cuh"

namespace texture_save_detail
{
    __global__ void convertGrayscaleToRgbKernel(const CudaTexture<uint8_t> *r_texture, uint8_t *w_texture)
    {
        const unsigned int invokeIndex = getInvokeIndex();
        if (!isWithinBounds(invokeIndex, r_texture->size()))
            return;

        const uint8_t val = (*r_texture)[invokeIndex];
        uint8_t red, green, blue;
        if (val != 255)
        {
            // Generate relatively high-contrast HSV color
            float h = 0.125f * static_cast<float>(val % 8) + static_cast<float>(val) / 8 / 256.0f;
            float s = 0.5f + 0.5f * sin(static_cast<float>(val) * 0.1f); // Oscillating between 0.0 and 1.0
            float v = 0.5f + 0.5f * cos(static_cast<float>(val) * 0.1f); // Oscillating between 0.0 and 1.0

            float c = v * s;
            float x = c * (1.0f - abs(fmod(h * 6.0f, 2.0f) - 1.0f));
            float m = v - c;

            float r, g, b;

            if (h >= 0.0 && h < 1.0 / 6.0)
            {
                r = c;
                g = x;
                b = 0.0f;
            } else if (h >= 1.0 / 6.0 && h < 2.0 / 6.0)
            {
                r = x;
                g = c;
                b = 0.0f;
            } else if (h >= 2.0 / 6.0 && h < 3.0 / 6.0)
            {
                r = 0.0f;
                g = c;
                b = x;
            } else if (h >= 3.0 / 6.0 && h < 4.0 / 6.0)
            {
                r = 0.0f;
                g = x;
                b = c;
            } else if (h >= 4.0 / 6.0 && h < 5.0 / 6.0)
            {
                r = x;
                g = 0.0f;
                b = c;
            } else
            {
                r = c;
                g = 0.0f;
                b = x;
            }

            red = static_cast<uint8_t>(std::lround((r + m) * 255.0f));
            green = static_cast<uint8_t>(std::lround((g + m) * 255.0f));
            blue = static_cast<uint8_t>(std::lround((b + m) * 255.0f));
        }
        else
        {
            // A slightly warmer background color to stand out of the white PDF background.
            red = 230;
            green = 235;
            blue = 240;
        }


        w_texture[invokeIndex * 3] = red;
        w_texture[invokeIndex * 3 + 1] = green;
        w_texture[invokeIndex * 3 + 2] = blue;
    }

    template<typename T>
    void saveRgbTextureToDisk(const char *fileName, T *imageData, int width, int height)
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
                     PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png_ptr, info_ptr);

        std::vector<png_bytep> row_pointers(height);
        for (int y = 0; y < height; y++)
            row_pointers[y] = reinterpret_cast<png_bytep>(&imageData[y * width * 3]);

        png_write_image(png_ptr, row_pointers.data());
        png_write_end(png_ptr, nullptr);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
    }
}

void saveGrayscale8BitCudaTextureToDiskAsRgb(const char *fileName, const CudaTextureHost<uint8_t> &texture)
{
    const int width = texture.width();
    const int height = texture.height();

    uint8_t *imageDataDevice;
    if (const cudaError_t err = cudaMalloc(&imageDataDevice, sizeof(uint8_t) * width * height * 3); err != cudaSuccess)
    {
        std::cerr << "Error malloc device memory during texture save: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Fixed 256 threads for now, should be fine
    int numBlocksPixels = (width * height + 1) / 256;

    texture_save_detail::convertGrayscaleToRgbKernel<<<numBlocksPixels, 256>>>(texture.deviceTexture(), imageDataDevice);

    std::vector<uint8_t> imageDataHost(width * height * 3);

    if (const cudaError_t err = cudaMemcpy(imageDataHost.data(), imageDataDevice, sizeof(uint8_t) * width * height * 3,
                                           cudaMemcpyDeviceToHost); err != cudaSuccess)
        std::cerr << "Error copy texture during disk save: " << cudaGetErrorString(err) << std::endl;

    if (const cudaError_t err = cudaFree(imageDataDevice); err != cudaSuccess)
        std::cerr << "Error freeing device memory during texture save: " << cudaGetErrorString(err) << std::endl;

    texture_save_detail::saveRgbTextureToDisk(fileName, imageDataHost.data(), width, height);
}
