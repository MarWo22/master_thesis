//
// Created by marti on 21/03/2025.
//

#ifndef BACKGROUND_H
#define BACKGROUND_H
#include "camera.h"
#include "render_component.h"
#include "shader.h"
#include "GL/glew.h"
#include "textures/texture.h"


class Background final : public RenderComponent{

    GLuint m_VAO;

    Shader *m_shader;
    Texture *m_environmentMap;

    float m_environmentMultiplier;
public:
    Background();
    void init() override;
    void render(const Camera &camera) override;

private:
    void initVAO();

};



#endif //BACKGROUND_H
