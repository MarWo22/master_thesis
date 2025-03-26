#ifndef RENDERER_H
#define RENDERER_H
#include <chrono>
#include <unordered_map>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "gui.h"
#include "render_component.h"
#include "shader.h"
#include "textures/texture.h"

class Renderer
{
    static std::unordered_map<std::string, Texture*> m_textures;
    static std::unordered_map<std::string, Shader*> m_shaders;
    std::vector<RenderComponent*> m_renderComponents;

    Camera m_camera;
    GLFWwindow *m_window;
    Gui m_gui;

    std::chrono::time_point<std::chrono::steady_clock> m_previousTime;
    float m_deltaTime;

public:
    Renderer();

    ~Renderer();

    void initRenderer();
    void initRenderComponents() const;

    void render();

    void renderRenderComponents() const;

    void updateTime();

    static std::unordered_map<std::string, Texture*> const &getTextures();
    static std::unordered_map<std::string, Shader*> const &getShaders();

    static void addTexture(const std::string &identifier, Texture *texture);
    static void addShader(const std::string &identifier, Shader *shader);

    void addRenderComponent(RenderComponent *renderComponent);
    [[nodiscard]] bool shouldClose() const;

private:
    void initCamera();
    void initWindow();
    void initInputHandling();
    void initImGUI() const;

    void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);

    void mouseCallback(GLFWwindow *window, double xPos, double yPos);
};


#endif //RENDERER_H
