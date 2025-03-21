#ifndef MIPMAP_H_
#define MIPMAP_H_

#include "GL/glew.h"
#include <string>
#include <vector>

class Mipmap
{
    GLuint id_ = 0;

public:
    Mipmap() = default;

    explicit Mipmap(const std::vector<std::string>& filenames);

    Mipmap(const std::string &base_path, const std::string &file_extension, int n_levels);


    void Load(const std::vector<std::string>& filenames);

    void Load(const std::string &base_path, const std::string &file_extension, int n_levels);

    [[nodiscard]] GLuint id() const;

    void Bind(GLenum texture_unit) const;

};


#endif //MIPMAP_H_
