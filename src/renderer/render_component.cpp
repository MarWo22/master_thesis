//
// Created by marti on 21/03/2025.
//

#include "render_component.h"

#include <iostream>

#include "renderer.h"


RenderComponent::~RenderComponent() = default;

void RenderComponent::init() {}

void RenderComponent::preRender() {}

void RenderComponent::postRender() {}

void RenderComponent::render(const Camera &camera) {}

Shader *RenderComponent::getShader(const std::string &identifier)
{
    auto shader_map = Renderer::getShaders();
    auto shaderIt = shader_map.find(identifier);
    if (shaderIt != shader_map.end()) {
        return shaderIt->second;  // Assign the address of the Shader object
    }
    std::cerr << "Shader 'background' not found!" << std::endl;
    return nullptr;
}
Texture *RenderComponent::getTexture(const std::string &identifier)
{
    auto texture_map = Renderer::getTextures();
    auto shaderIt = texture_map.find(identifier);
    if (shaderIt != texture_map.end()) {
        return shaderIt->second;  // Assign the address of the Shader object
    }
    std::cerr << "Shader 'background' not found!" << std::endl;
    return nullptr;
}
