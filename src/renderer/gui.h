#ifndef GUI_H
#define GUI_H

class Gui {

    float m_frameCount;
    float m_avgFramerate;

    float m_timeSum;
public:
    Gui();
    ~Gui();
    void gui();


    void render();

    static void preRenderGUI();
};



#endif //GUI_H
