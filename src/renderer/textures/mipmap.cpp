//
// Created by marti on 23/04/2024.
//

#include <iostream>
#include "mipmap.h"
#include "image.h"

Mipmap::Mipmap(const std::vector<std::string>& filenames)
{
    Load(filenames);
}

Mipmap::Mipmap(const std::string &base_path, const std::string &file_extension, const int n_levels)
{
    Load(base_path, file_extension, n_levels);
}

void Mipmap::Load(const std::vector<std::string>& filenames)
{
    glGenTextures(1, &id_);
    glBindTexture(GL_TEXTURE_2D, id_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    const Image image(filenames[0], 3);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, image.width, image.height, 0, GL_RGB, GL_FLOAT, image.data);
    glGenerateMipmap(GL_TEXTURE_2D);

// We call this again because AMD drivers have some weird issue in the GenerateMipmap function that
// breaks the first level of the image.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, image.width, image.height, 0, GL_RGB, GL_FLOAT, image.data);
    for(int i = 1; i < filenames.size(); i++)
    {
        const auto im = Image(filenames[i], 3);

        glTexImage2D(GL_TEXTURE_2D, i, GL_RGB32F, im.width, im.height, 0, GL_RGB, GL_FLOAT, im.data);

    }
}

void Mipmap::Load(const std::string &base_path, const std::string &file_extension, const int n_levels)
{
    std::vector<std::string> filenames;

    for (int idx = 0; idx < n_levels; ++idx)
        filenames.push_back(std::string(base_path).append(std::to_string(idx)).append(file_extension));

    Load(filenames);
}


GLuint Mipmap::id() const
{
    return id_;
}

void Mipmap::Bind(const GLenum texture_unit) const
{
    glActiveTexture(texture_unit);
    glBindTexture(GL_TEXTURE_2D, id_);
}
