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
    , m_timeSum(0)
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

    if (ImGui::CollapsingHeader("Render mode"))
    {
        if (ImGui::RadioButton("Normal", &generationSettings.renderMode, GenerationSettings::RenderMode::NORMAL))
        {
            generationSettings.callCallback("toggleDefaultMode");
        }
        if (ImGui::RadioButton("Show Plates", &generationSettings.renderMode, GenerationSettings::RenderMode::SHOW_PLATES))
        {
            generationSettings.callCallback("togglePlateMode");
        }
        if (ImGui::RadioButton("Show Collision Areas (requires a new iteration)", &generationSettings.renderMode, GenerationSettings::RenderMode::SHOW_COLLISION_AREAS))
        {
            generationSettings.callCallback("toggleCollisionMode");
        }
        if (ImGui::RadioButton("Show Plate Directions", &generationSettings.renderMode, GenerationSettings::RenderMode::SHOW_PLATE_DIRECTIONS))
        {
            generationSettings.callCallback("toggleDirectionMode");
        }
        if (ImGui::RadioButton("Show Plate Velocities", &generationSettings.renderMode, GenerationSettings::RenderMode::SHOW_PLATE_VELOCITIES))
        {
            generationSettings.callCallback("toggleVelocityMode");
        }
        if (ImGui::RadioButton("Show Uplift Areas", &generationSettings.renderMode, GenerationSettings::RenderMode::SHOW_UPLIFT_AREAS))
        {
            generationSettings.callCallback("toggleUpliftMode");
        }
    }

    ImGui::SetNextItemWidth(250);
    ImGui::InputInt("Number of iterations", &generationSettings.executionIterations);

    bool disable_button = generationSettings.isExecutingRealtime;

    if (disable_button)
        ImGui::BeginDisabled(); // disables all widgets until EndDisabled is called

    if (ImGui::Button("Execute Realtime"))
    {
        generationSettings.isExecutingRealtime = true;
        generationSettings.callCallback("executeIterations");
        m_timeSum = 0;
    }

    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);

    ImGui::InputInt("it/s", &generationSettings.iterationsPerSecond);

    if (disable_button)
        ImGui::EndDisabled();

    if (ImGui::Button("Execute At Once"))
    {
        generationSettings.isExecutingRealtime = false;
        generationSettings.callCallback("executeIterations");
    }

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


