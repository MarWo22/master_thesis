#ifndef GL_MISC_H
#define GL_MISC_H

#include <string>

#define CHECK_GL_ERROR()                                                                                     \
	{                                                                                                        \
		gl_misc::checkGLError(__FILE__, __LINE__) && (__debugbreak(), 1);                                    \
	}

namespace gl_misc
{
    void non_fatal_error(std::string errorString, std::string title);

    void fatal_error(std::string errorString, std::string title);

    bool checkGLError(const char* file, int line);

    void startupGLDiagnostics();

    void setupGLDebugMessages();


};


#endif //GL_MISC_H
