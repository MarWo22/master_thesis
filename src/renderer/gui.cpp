//
// Created by marti on 21/03/2025.
//

#include "gui.h"

#include <chrono>
#include <iostream>
#include <random>

#include "../generation_settings.h"
#include "../cuda_compute/kernel_settings.cuh"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include <nfd.h>
#include <filesystem>

#include "../cuda_compute/texture_manager.cuh"


RenderSettings renderSettings;
SimulationSettings simulationSettings;
SaveTextureGui saveTextureGui;

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
            if (ImGui::RadioButton("Normal", &renderSettings.renderMode, RenderSettings::RenderMode::NORMAL))
            {
                renderSettings.callCallback("renderSettingsChanged");
            }
            if (ImGui::RadioButton("Show Collision Areas (requires a new iteration)", &renderSettings.renderMode,
                                   RenderSettings::RenderMode::SHOW_COLLISION_AREAS))
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
        }
        if (ImGui::CollapsingHeader("Render Options"))
        {
            if (ImGui::Checkbox("Displace Height", &renderSettings.renderHeight))
                renderSettings.callCallback("renderSettingsChanged");

            if (ImGui::Checkbox("Show Plate Borders", &renderSettings.renderBorders))
                renderSettings.callCallback("renderSettingsChanged");

            if (ImGui::Checkbox("Show Water", &renderSettings.renderWater))
                renderSettings.callCallback("renderSettingsChanged");

            if (ImGui::Checkbox("Show Plate Directions", &renderSettings.renderDirections))
                renderSettings.callCallback("renderSettingsChanged");

            if (ImGui::Checkbox("Show Stress Peaks", &renderSettings.renderStressPeaks))
                renderSettings.callCallback("renderSettingsChanged");



            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Height Displacement Multiplier", &renderSettings.heightMultiplier, 0.1f, 0.f, 500.f);
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
            // e.g. ImGui Slider to update newSettings
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Inelastic Collision Multiplier", &newSettings.inelasticCollisionMultiplier, 0.005f, 0.f,
                             20.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Merge Direction Dot Min Threshold", &newSettings.mergeDotDirectionThreshold, 0.001f, 0.0f,
                             1.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Merge Velocity Diff Max Threshold", &newSettings.mergeVelocityDiffThreshold, 0.001f, 0.f,
                             1.f);
            ImGui::SetNextItemWidth(250);
            ImGui::DragInt("Min Plate Size", &newSettings.minPlateSize, 1, 1, 100);

            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Divergence Height Target", &newSettings.divergence_height_target, 0.1f, 1, 250);

            ImGui::SetNextItemWidth(250);
            ImGui::DragFloat("Divergence Interpolation Factor", &newSettings.divergence_interpolation_factor, 0.001f, 0.f, 1.f);
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
            ImGui::DragFloat("Thermal Threshold Angle", &newSettings.thermalThresholdAngle, 0.01f, 0.f, 2.f);
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
