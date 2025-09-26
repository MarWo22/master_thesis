//
// Created by marti on 21/03/2025.
//

#include "gui.h"

#include <array>
#include <chrono>
#include <iostream>
#include <random>

#include "../generation_settings.h"
#include "../cuda_compute/kernel_settings.cuh"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include <nfd.h>
#include <filesystem>
#include <glm/common.hpp>
#include <glm/vec3.hpp>

#include "../cuda_compute/plate_tectonics_kernel.cuh"
#include "../cuda_compute/texture_manager.cuh"


RenderSettings renderSettings;
SimulationSettings simulationSettings;
SaveTextureGui saveTextureGui;

std::array<GuiPlateData, MAX_PLATE_COUNT> guiPlateData;


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

    const size_t usedMemory = TextureManager::getAllocatedMemory();
    ImGui::Text("Allocated GPU memory %.1f MB", usedMemory / 1000000.f);

    renderSettingsSection();
    simulationSettingsSection();
    resetSimulationSection();
    saveTextureSection();
    executionSettingsSection();


    static bool render_with_transparency = false;

    if (render_with_transparency)
        ImGui::Begin("Plate ID Legend", nullptr, ImGuiWindowFlags_NoBackground);
    else
        ImGui::Begin("Plate ID Legend", nullptr);

    // Scrollable child region with fixed 300px height

    ImGui::BeginChild(
        "LegendScrollRegion",
        ImVec2(0, 0),
        false,
        ImGuiWindowFlags_AlwaysUseWindowPadding |
        ImGuiWindowFlags_HorizontalScrollbar |
        ImGuiWindowFlags_AlwaysVerticalScrollbar
    );

    bool pushed = false;
    if (render_with_transparency)
    {
        pushed = true;
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 0, 0, 255)); // Black text
    }

    ImGui::Checkbox("Transparent window background", &render_with_transparency);


    if (ImGui::Checkbox("Show Plate Info", &renderSettings.copyPlateData))
        if (renderSettings.copyPlateData)
            renderSettings.callCallback("copyPlateInfo");

    ImGui::Spacing();

    for (int id = 0; id < MAX_PLATE_COUNT; ++id)
    {
        if (!renderSettings.copyPlateData || guiPlateData[id].size > 0)
        {
            glm::vec3 color = getColorFromID(id);
            ImVec4 imColor(color.r, color.g, color.b, 1.0f);

            ImGui::ColorButton(("##color" + std::to_string(id)).c_str(), imColor,
                               ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoBorder,
                               ImVec2(20, 20));

            ImGui::SameLine();
            ImGui::Text("Plate ID: %d", id);
        }
        if (renderSettings.copyPlateData && guiPlateData[id].size > 0)
        {
            ImGui::Text("Mass: %.3f", guiPlateData[id].mass);
            ImGui::Text("Size: %d", guiPlateData[id].size);
            ImGui::Text("Has moved: %s", guiPlateData[id].hasMoved ? "true" : "false");
            ImGui::Text("Velocity: %.3f", guiPlateData[id].velocity);
            ImGui::Text("Direction: (%.3f, %.3f)", guiPlateData[id].direction.x, guiPlateData[id].direction.y);

            std::ostringstream continental_ss;
            for (int i = 0; i < guiPlateData[id].continental_len; ++i)
            {
                if (i > 0) continental_ss << ", ";
                continental_ss << guiPlateData[id].continental[i];
            }
            std::string continental = continental_ss.str();

            std::ostringstream subductions_ss;
            for (int i = 0; i < guiPlateData[id].subduction_len; ++i)
            {
                if (i > 0) subductions_ss << ", ";
                subductions_ss << guiPlateData[id].subductions[i];
            }
            std::string subductions = subductions_ss.str();

            ImGui::Text("Continental collisions: (%s)", continental.c_str());
            ImGui::Text("Subductions collisions: (%s)", subductions.c_str());

            ImGui::Text("Perimeter: (%i)", guiPlateData[id].perimeter);
            ImGui::Text("Circularity: (%.2f)", guiPlateData[id].circularity);
            ImGui::Text("Break score: (%.2f)", guiPlateData[id].break_score);

        }
    }

    if (pushed)
        ImGui::PopStyleColor();

    ImGui::EndChild();

    ImGui::End();
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

void Gui::renderSettingsSection()
{
    if (ImGui::CollapsingHeader("Render Settings"))
    {
        ImGui::Indent(15);
        if (ImGui::CollapsingHeader("Render mode"))
        {
            ImGui::Indent(15);
            if (ImGui::RadioButton("Normal", &renderSettings.renderMode, RenderSettings::RenderMode::NORMAL))
            {
                renderSettings.callCallback("renderSettingsChanged");
            }
            if (ImGui::RadioButton("Show Plate Velocities", &renderSettings.renderMode,
                                   RenderSettings::RenderMode::SHOW_PLATE_VELOCITIES))
            {
                renderSettings.callCallback("renderSettingsChanged");
            }
            if (ImGui::RadioButton("Show Pressure Areas", &renderSettings.renderMode,
                                   RenderSettings::RenderMode::SHOW_PRESSURE_AREAS))
            {
                renderSettings.callCallback("renderSettingsChanged");
            }
            if (ImGui::RadioButton("Show Stress Areas", &renderSettings.renderMode,
                                   RenderSettings::RenderMode::SHOW_STRESS_AREAS))
            {
                renderSettings.callCallback("renderSettingsChanged");
            }
            ImGui::Unindent(15.0f);
        }
        if (ImGui::CollapsingHeader("Normal Render Options"))
        {
            ImGui::Indent(15);
            if (ImGui::CollapsingHeader("Shading"))
            {
                ImGui::Indent(15);
                if (ImGui::RadioButton("Normal Shading", &renderSettings.shadingMode,
                                       RenderSettings::ShadingMode::NORMAL_SHADING))
                {
                    renderSettings.callCallback("renderSettingsChanged");
                }
                if (ImGui::RadioButton("Show Crust Type", &renderSettings.shadingMode,
                                       RenderSettings::ShadingMode::SHOW_CRUST_TYPE))
                {
                    renderSettings.callCallback("renderSettingsChanged");
                }
                if (ImGui::RadioButton("Show Plate Ids", &renderSettings.shadingMode,
                                       RenderSettings::ShadingMode::SHOW_PLATE_IDS))
                {
                    renderSettings.callCallback("renderSettingsChanged");
                }
                ImGui::Unindent(15.0f);
            }
            if (ImGui::CollapsingHeader("Plate Borders"))
            {
                ImGui::Indent(15);
                if (ImGui::RadioButton("No Borders", &renderSettings.borderRenderMode,
                                       RenderSettings::BorderRenderMode::NO_BORDER))
                    renderSettings.callCallback("renderSettingsChanged");
                if (ImGui::RadioButton("Render Smooth Borders", &renderSettings.borderRenderMode,
                                       RenderSettings::BorderRenderMode::SMOOTH_BORDER))
                    renderSettings.callCallback("renderSettingsChanged");
                if (ImGui::RadioButton("Render Raw Borders", &renderSettings.borderRenderMode,
                                       RenderSettings::BorderRenderMode::RAW_BORDER))
                    renderSettings.callCallback("renderSettingsChanged");
                if (ImGui::RadioButton("Render Collision Borders", &renderSettings.borderRenderMode,
                                       RenderSettings::BorderRenderMode::COLLISIONS))
                    renderSettings.callCallback("renderSettingsChanged");
                ImGui::Unindent(15.0f);
            }

            if (ImGui::Checkbox("Displace Height", &renderSettings.renderHeight))
                renderSettings.callCallback("renderSettingsChanged");

            if (ImGui::Checkbox("Show Water", &renderSettings.renderWater))
                renderSettings.callCallback("renderSettingsChanged");

            if (ImGui::Checkbox("Show Plate Directions", &renderSettings.renderDirections))
                renderSettings.callCallback("renderSettingsChanged");

            if (ImGui::Checkbox("Show Stress Peaks", &renderSettings.renderStressPeaks))
                renderSettings.callCallback("renderSettingsChanged");


            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Height Displacement Multiplier", &renderSettings.heightMultiplier, 0.1f, 0.f, 500.f);

            ImGui::Unindent(15.0f);
        }
        ImGui::Unindent(15.0f);
    }
}

void Gui::simulationSettingsSection()
{
    KernelSettings newSettings = kernelSettingsHost; // start from old values

    if (ImGui::CollapsingHeader("Simulation Settings"))
    {
        ImGui::Indent(15);
        if (ImGui::CollapsingHeader("Plate Properties"))
        {
            if (ImGui::CollapsingHeader("Collision Momentum Properties"))
            {
                ImGui::Indent(15);

                // e.g. ImGui Slider to update newSettings
                ImGui::SetNextItemWidth(250);
                ImGui::DragFloat("Inelastic Subduction Collision Multiplier",
                                 &newSettings.inelasticCollisionMultiplierSubduction, 0.0005f, 0.f,
                                 1.f);
                ImGui::SetNextItemWidth(250);
                ImGui::DragFloat("Inelastic Continental Collision Multiplier",
                                 &newSettings.inelasticCollisionMultiplierContinental, 0.0005f, 0.f,
                                 1.f);
                ImGui::SetNextItemWidth(250);
                ImGui::DragFloat("Subduction Friction Coefficient", &newSettings.frictionCoefficientSubduction, 0.0005f,
                                 0.f,
                                 5.f);
                ImGui::SetNextItemWidth(250);
                ImGui::DragFloat("Continental Friction Coefficient", &newSettings.frictionCoefficientContinental,
                                 0.0005f, 0.f,
                                 5.f);
                ImGui::SetNextItemWidth(250);
                ImGui::DragFloat("Environmental Drag Coefficient", &newSettings.environmentalDragCoefficient, 0.0001,
                                 0.f, 1.f, "%.4f");

                ImGui::Unindent(15.0f);
            }

            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Continental Crust Threshold", &newSettings.continentalCrustThreshold, 0.05f, 1.f, 250.f);

            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Merge Direction Dot Min Threshold", &newSettings.mergeDotDirectionThreshold, 0.001f, 0.0f,
                             1.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Merge Velocity Diff Max Threshold", &newSettings.mergeVelocityDiffThreshold, 0.001f, 0.f,
                             1.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragInt("Min Plate Size", &newSettings.minPlateSize, 1, 1, 100);

            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Divergence Max Height Target", &newSettings.divergence_height_target, 0.1f, 1, 250);

            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Divergence Interpolation Factor", &newSettings.divergence_interpolation_factor, 0.001f,
                             0.f, 1.f);
            // direction_threshold_merge
            // velocity_threshold_merge
            // min_size
        }

        if (ImGui::CollapsingHeader("Uplift Properties"))
        {
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Uplift multiplier", &newSettings.upliftMultiplier, 0.01f, 0.01f, 50.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragInt("Uplift range", &newSettings.upliftRange, 1, 0, 100);
        }

        if (ImGui::CollapsingHeader("Erosion Properties"))
        {
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Water pipe cross section", &newSettings.hydrationPipeCrossSection, 0.005f, 0.f, 50.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Water pipe length", &newSettings.hydrationPipeLength, 0.005f, 0.f, 500.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Rainfall", &newSettings.hydrationRainfall, 0.005f, 0.f, 500.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Evaporation", &newSettings.hydrationEvaporation, 0.005f, 0.f, 500.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Minimum Water Level", &newSettings.minimumWaterLevel, 1.0f, 0.f, 500.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Capacity", &newSettings.sedimentCapacity, 0.05f, 0.f, 10.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Dissolving constant", &newSettings.sedimentDissolving, 0.0001f, 0.f, 10.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Thermal Erosion Strength", &newSettings.thermalErosionStrength, 0.001f, 0.f, 2.0f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Thermal Erosion Amplitude", &newSettings.thermalErosionAmplitude, 0.01f, 0.f, 2.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Thermal Cell Size", &newSettings.thermalCellSize, 0.1f, 0.1f, 5.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Thermal Threshold Angle", &newSettings.thermalThresholdAngle, 0.01f, 0.01f, 2.f);
        }

        if (ImGui::CollapsingHeader("Pressure Properties"))
        {
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Pressure Accumulation Rate", &newSettings.pressureAccumulation, 0.001f, 0.f, 0.1f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Pressure Decay Rate", &newSettings.pressureDecayRate, 0.01f, 0.f, 1.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragInt("Pressure Blur Range", &newSettings.pressureBlurRange, 1, 1, 100);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Pressure Multiplier", &newSettings.pressureMultiplier, 0.01f, 0.f, 10.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Stress Split Threshold", &newSettings.stressSplitThreshold, 1.0f, 1.0f, 200.0f);
        }

        ImGui::Unindent(15.0f);
    }

    // If changed, copy to device and update host copy
    if (newSettings != kernelSettingsHost)
    {
        kernelSettingsHost = newSettings;

        // // Copy to device constant memory
        copyHostKernelSettingsToDevice();
    }
}

void Gui::saveTextureSection()
{
    if (ImGui::CollapsingHeader("Save Textures"))
    {
        ImGui::Indent(15);

        ImGui::RadioButton("Heightmap", &saveTextureGui.texture, SaveTextureGui::TextureType::HEIGHTMAP);
        ImGui::RadioButton("Plate IDs", &saveTextureGui.texture, SaveTextureGui::TextureType::PLATE_IDS);

        if (ImGui::Button("Save"))
        {
            nfdchar_t *outPath = nullptr;
            constexpr nfdfilteritem_t filterItem[1] = {{"PNG files", "png"}};

            const std::string cwd = std::filesystem::current_path().string();
            if (const nfdresult_t result = NFD_SaveDialog(&outPath, filterItem, 1, cwd.c_str(), "texture.png");
                result == NFD_OKAY)
            {
                saveTextureGui.path = outPath;
                free(outPath); // Always free the result
                if (saveTextureGui.saveTextureCallback)
                    saveTextureGui.saveTextureCallback();
                else
                    std::cerr << "No saving callback set\n";
            }
        }

        ImGui::Unindent(15.0f);
    }
}

void Gui::executionSettingsSection()
{
    ImGui::SetNextItemOpen(true, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Execution"))
    {
        ImGui::SetNextItemWidth(250);
        ImGui::InputInt("Number of iterations", &renderSettings.executionIterations);

        bool disable_button = renderSettings.isExecutingRealtime;

        if (disable_button)
            ImGui::BeginDisabled(); // disables all widgets until EndDisabled is called

        if (ImGui::Button("Execute Realtime"))
        {
            renderSettings.isExecutingRealtime = true;
            renderSettings.callCallback("executeIterations");
            m_timeSum = 0;
        }

        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);

        ImGui::InputInt("it/s", &renderSettings.iterationsPerSecond);

        if (disable_button)
            ImGui::EndDisabled();

        if (ImGui::Button("Execute At Once"))
        {
            renderSettings.isExecutingRealtime = false;
            renderSettings.callCallback("executeIterations");
        }
    }
}

void Gui::resetSimulationSection()
{
    if (ImGui::CollapsingHeader("Reset Simulation"))
    {
        ImGui::SetNextItemWidth(250);
        ImGui::InputScalar("Seed", ImGuiDataType_U32, &simulationSettings.seed);

        if (ImGui::Button("Generate New Seed"))
            simulationSettings.seed = generateRandomSeed();


        ImGui::SetNextItemWidth(250);
        ImGui::InputInt("Number of starting plates", &simulationSettings.numStartingPlates);

        static bool gen1Enabled = true;
        static bool gen2Enabled = true;

        ImGui::Checkbox("##enableGen1", &gen1Enabled);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(250);
        ImGui::InputInt("Number of Voronoi seeds Gen1", &simulationSettings.numVoronoiSeeds[0]);

        ImGui::Checkbox("##enableGen2", &gen2Enabled);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(250);
        ImGui::InputInt("Number of Voronoi seeds Gen2", &simulationSettings.numVoronoiSeeds[1]);

        if (ImGui::Button("Reset Simulation##1"))
        {
            std::vector<int> voronoiSeeds;
            if (gen1Enabled)
                voronoiSeeds.push_back(simulationSettings.numVoronoiSeeds[0]);
            if (gen2Enabled)
                voronoiSeeds.push_back(simulationSettings.numVoronoiSeeds[1]);

            if (simulationSettings.resetCallback)
                simulationSettings.resetCallback(simulationSettings.seed, simulationSettings.numStartingPlates,
                                                 voronoiSeeds);
            else
                std::cerr << "No reset callback registered\n";
        }
    }
}

unsigned int Gui::generateRandomSeed()
{
    std::random_device rd;
    const auto time_now = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    const std::seed_seq seed_seq{
        rd(), static_cast<unsigned int>(time_now & 0xFFFFFFFF), static_cast<unsigned int>((time_now >> 32) & 0xFFFFFFFF)
    };
    std::vector<unsigned int> seeds(1);
    seed_seq.generate(seeds.begin(), seeds.end());

    return static_cast<int>(seeds[0]);
}

// Function to convert HSV to RGB
glm::vec3 Gui::hsvToRgb(const float h, const float s, const float v)
{
    const float c = v * s;
    const float x = c * (1.0f - std::fabs(fmod(h * 6.0f, 2.0f) - 1.0f));
    const float m = v - c;

    float r = 0, g = 0, b = 0;

    if (h >= 0.0f && h < 1.0f / 6.0f)
    {
        r = c;
        g = x;
        b = 0.0f;
    } else if (h >= 1.0f / 6.0f && h < 2.0f / 6.0f)
    {
        r = x;
        g = c;
        b = 0.0f;
    } else if (h >= 2.0f / 6.0f && h < 3.0f / 6.0f)
    {
        r = 0.0f;
        g = c;
        b = x;
    } else if (h >= 3.0f / 6.0f && h < 4.0f / 6.0f)
    {
        r = 0.0f;
        g = x;
        b = c;
    } else if (h >= 4.0f / 6.0f && h < 5.0f / 6.0f)
    {
        r = x;
        g = 0.0f;
        b = c;
    } else
    {
        r = c;
        g = 0.0f;
        b = x;
    }

    return {r + m, g + m, b + m};
}


glm::vec3 Gui::getColorFromID(const int id)
{
    // Generate a hue that cycles every 8 IDs, with slow hue shift after that
    const float hue = 0.125f * (id % 8) + static_cast<float>(id / 8) / 256.0f;

    // Oscillate saturation and value for variation
    const float saturation = 0.5f + 0.5f * std::sin(id * 0.1f);
    const float value = 0.5f + 0.5f * std::cos(id * 0.1f);

    return hsvToRgb(hue, saturation, value);
}
