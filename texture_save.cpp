#include "texture_save.h"

#include <png.h>
#include <vector>


namespace
{
    template <typename T>
    void saveGrayscalePng(const char *fileName, T *imageData, const int width, const int height, const int bitDepth)
    {
        FILE *fp;

        if (const errno_t err = fopen_s(&fp, fileName, "wb"); err != 0)
        {
            std::cerr << "Error: Cannot open file '" << fileName << "' : " << err << std::endl;
            return;
        }

        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png_ptr)
        {
            std::cerr << "Error: Cannot create PNG write struct" << std::endl;
            fclose(fp);
            return;
        }

        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr)
        {
            std::cerr << "Error: Cannot create PNG info struct" << std::endl;
            fclose(fp);
            return;
        }

        if (setjmp(png_jmpbuf(png_ptr)))
        {
            std::cerr << "Error: Writing png failed" << std::endl;
            png_destroy_write_struct(&png_ptr, &info_ptr);
            fclose(fp);
            return;
        }

        png_init_io(png_ptr, fp);

        png_set_IHDR(png_ptr, info_ptr, width, height, bitDepth, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        png_write_info(png_ptr, info_ptr);

        // Write row pointers
        std::vector<png_bytep> row_pointers(height);
        for (int y = 0; y < height; y++)
            row_pointers[y] = reinterpret_cast<png_bytep>(&imageData[y * width]);

        png_write_image(png_ptr, row_pointers.data());
        png_write_end(png_ptr, nullptr);

        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        std::cout << "Saved " << fileName << " successfully!\n";
    }

    template <typename T>
    void saveRgbPng(const char *fileName, T *imageData, const int width, const int height, const int bitDepth)
    {
        FILE *fp;
        if (const errno_t err = fopen_s(&fp, fileName, "wb"); err != 0)
        {
            std::cerr << "Error: Cannot open file '" << fileName << "' : " << err << std::endl;
            return;
        }

        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png_ptr)
        {
            std::cerr << "Error: Cannot create PNG write struct" << std::endl;
            fclose(fp);
            return;
        }

        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr)
        {
            std::cerr << "Error: Cannot create PNG info struct" << std::endl;
            png_destroy_write_struct(&png_ptr, nullptr);
            fclose(fp);
            return;
        }

        if (setjmp(png_jmpbuf(png_ptr)))
        {
            std::cerr << "Error: Writing PNG failed" << std::endl;
            png_destroy_write_struct(&png_ptr, &info_ptr);
            fclose(fp);
            return;
        }

        png_init_io(png_ptr, fp);
        png_set_IHDR(png_ptr, info_ptr, width, height, bitDepth, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png_ptr, info_ptr);

        // Write row pointers
        std::vector<png_bytep> row_pointers(height);
        for (int y = 0; y < height; y++)
            row_pointers[y] = reinterpret_cast<png_bytep>(&imageData[y * width * 3]);

        png_write_image(png_ptr, row_pointers.data());
        png_write_end(png_ptr, nullptr);

        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        std::cout << "Saved " << fileName << " successfully!\n";
    }

}


void Texture::save16BitGreyscalePng(const char *fileName, float *image_data, int width, int height)
{
    saveGrayscalePng(fileName, image_data, width, height, 16);
}

void Texture::save16BitGreyscalePng(const char *fileName, uint16_t *image_data, const int width, const int height)
{
    saveGrayscalePng<uint16_t>(fileName, image_data, width, height, 16);
}
void Texture::save8BitGreyscalePng(const char *fileName, uint8_t *image_data, const int width, const int height)
{
    saveGrayscalePng(fileName, image_data, width, height, 8);
}

void Texture::save8BitRgbPng(const char *fileName, uint8_t *image_data, const int width, const int height)
{
    saveRgbPng(fileName, image_data, width, height, 8);
}
