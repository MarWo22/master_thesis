#ifndef GUI_H
#define GUI_H

class Gui {

    float m_frameCount;
    float m_avgFramerate;

    float m_timeSum;

public:
    Gui();
    ~Gui();

    void render();

private:
    void gui();
    static void preRenderGUI();

    static void renderSettingsSection();
    static void simulationSettingsSection();
    static void saveTextureSection();
    void executionSettingsSection();

    static void resetSimulationSection();
    static unsigned int generateRandomSeed();
};




#endif //GUI_H
