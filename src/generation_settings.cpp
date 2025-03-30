#include "generation_settings.h"

#include <iostream>


void GenerationSettings::registerCallback(const std::string &ident, const std::function<void()> &callback)

{
    if (!m_callbacks.contains(ident))
        m_callbacks[ident] = std::vector<std::function<void()>>();

    m_callbacks[ident].push_back(callback);
}
void GenerationSettings::callCallback(const std::string &ident)
{

    if (m_callbacks.contains(ident))
        for (auto &callback : m_callbacks[ident])
            callback(); // Execute each callback
    else
        std::cerr << "No callback registered for " << ident << std::endl;
}
