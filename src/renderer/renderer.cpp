//
// Created by marti on 21/03/2025.
//

#include "renderer.h"
#include <GLFW/glfw3.h>

#include <chrono>
#include <imgui.h>
#include <iostream>

#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

std::unordered_map<std::string, Texture*> Renderer::m_textures = std::unordered_map<std::string, Texture*>();
std::unordered_map<std::string, Shader*> Renderer::m_shaders = std::unordered_map<std::string, Shader*>();

Renderer::Renderer()
    : m_camera()
    , m_window(nullptr)
    , m_deltaTime(0)
{}

Renderer::~Renderer()
{
    glfwTerminate();
}

void Renderer::initRenderer()
{
    initWindow();
    initInputHandling();
    initImGUI();
    initCamera();

    m_previousTime = std::chrono::high_resolution_clock::now();
}


void Renderer::initRenderComponents() const
{
    for (const auto renderComponent : m_renderComponents)
        renderComponent->init();
}

void Renderer::render()
{
    updateTime();

    int windowWidth, windowHeight;
    glfwGetWindowSize(m_window, &windowWidth, &windowHeight);
    glfwPollEvents();

        // Update the camera based of user input
    m_camera.Update(m_deltaTime, static_cast<float>(windowWidth) / static_cast<float>(windowHeight));
        // Start rendering cycle: Clear the color buffer

    glViewport(0, 0, windowWidth, windowHeight);
    glClearColor(0.2f, 0.2f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Do required ImGUI stuff
    renderRenderComponents();
    m_gui.render();

    glfwSwapBuffers(m_window);
}

void Renderer::renderRenderComponents() const
{
    for (const auto renderComponent : m_renderComponents)
        renderComponent->preRender();

    for (const auto renderComponent : m_renderComponents)
        renderComponent->render(m_camera);

    for (const auto renderComponent : m_renderComponents)
        renderComponent->postRender();
}

void Renderer::updateTime()
{
    const auto newTime = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float> timeDiff = newTime - m_previousTime;
    m_deltaTime = timeDiff.count();
    m_previousTime = newTime;
}

const std::unordered_map<std::string, Texture*> &Renderer::getTextures()
{
    return m_textures;
}
const std::unordered_map<std::string, Shader*> &Renderer::getShaders()
{
    return m_shaders;
}

void Renderer::addTexture(const std::string &identifier, Texture *texture)
{
    m_textures[identifier] = texture;
}

void Renderer::addShader(const std::string &identifier, Shader *shader)
{
    m_shaders[identifier] = shader;
}

void Renderer::addRenderComponent(RenderComponent *renderComponent)
{
    m_renderComponents.push_back(renderComponent);
}

bool Renderer::shouldClose() const
{
    return glfwWindowShouldClose(m_window);
}


void Renderer::initCamera()
{
    m_camera = Camera(glm::vec3(-70.0f, 50.0f, 70.0f), 45.f, 25.f, 5.f, 0.15f);
}

void Renderer::initWindow()
{
    /* Initialize the library */
    if (!glfwInit())
    {
        std::cerr << "Error initializing GLFW" << std::endl;
        exit(-1);
    }

    /* Create a windowed mode window and its OpenGL context */
    m_window = glfwCreateWindow(1280, 720, "Hello World", nullptr, nullptr);
    if (!m_window)
    {
        std::cerr << "Error initializing window" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    /* Make the window's context current */
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(0);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        glfwTerminate();
        exit(-1);
    }
}

void Renderer::initInputHandling()
{
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glfwSetKeyCallback(m_window, [](GLFWwindow* window, const int key, const int scancode, const int action, const int mods) {
        if (auto* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window))) {
            renderer->keyCallback(window, key, scancode, action, mods);
        }
    });

    glfwSetCursorPosCallback(m_window, [](GLFWwindow* window, const double xPos, const double yPos) {
        if (auto* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window))) {
            renderer->mouseCallback(window, xPos, yPos);
        }
    });

    glfwSetWindowUserPointer(m_window, this);
}

void Renderer::initImGUI() const
{
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init();
    ImGui::StyleColorsDark();
}

void Renderer::keyCallback(GLFWwindow* window, const int key, [[maybe_unused]] const int scancode, const int action, [[maybe_unused]] const int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL)
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        else
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
    m_camera.HandleKeyInput(key, action);
}

void Renderer::mouseCallback(GLFWwindow* window, const double xPos, const double yPos)
{
    const bool allowMovement = glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
    m_camera.HandleMouseInput(static_cast<float>(xPos), static_cast<float>(yPos), allowMovement);
}

