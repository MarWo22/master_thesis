//
// Created by marti on 21/03/2025.
//

#ifndef RENDERCOMPONENT_H
#define RENDERCOMPONENT_H
#include <string>
#include <unordered_map>

#include "camera.h"
#include "shader.h"
#include "textures/texture.h"


class RenderComponent {
public:

    virtual ~RenderComponent();

    virtual void init();
    virtual void preRender();
    virtual void postRender();
    virtual void render(const Camera &camera);

protected:
    static Shader *getShader(const std::string &identifier);
    static Texture *getTexture(const std::string &identifier);
};



#endif //RENDERCOMPONENT_H
