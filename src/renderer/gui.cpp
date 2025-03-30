//
// Created by marti on 21/03/2025.
//

#include "gui.h"

#include <iostream>

#include "../generation_settings.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

GenerationSettings generationSettings;

Gui::Gui()
    : m_frameCount(0)
    , m_avgFramerate(0)
{}

Gui::~Gui()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void Gui::gui()
{
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
              ImGui::GetIO().Framerate);

    m_avgFramerate = (m_avgFramerate * m_frameCount + ImGui::GetIO().Framerate) / (m_frameCount + 1);
    m_frameCount++;

    ImGui::Text("Average %.1f FPS", m_avgFramerate);

    ImGui::Text("Render mode:");
    ImGui::RadioButton("Normal", &generationSettings.renderMode, GenerationSettings::RenderMode::NORMAL);
    ImGui::RadioButton("Show Plates", &generationSettings.renderMode, GenerationSettings::RenderMode::SHOW_PLATES);

    ImGui::InputInt("Number of iterations", &generationSettings.executionIterations);

    if (ImGui::Button("Execute"))
        generationSettings.callCallback("executeIterations");
}

void Gui::render()
{
    preRenderGUI();
    gui();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Gui::preRenderGUI()
{
    ImGui_ImplGlfw_NewFrame();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();
}


