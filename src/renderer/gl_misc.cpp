#include <windows.h>
#include <iostream>
#include <sstream>
#include <debugapi.h>
#include <GL/glew.h>

#include "gl_misc.h"

// Error reporting
void gl_misc::fatal_error(std::string errorString, std::string title)
{
    if(title.empty())
    {
        title = "GL-Tutorial - Error";
    }
    if(errorString.empty())
    {
        errorString = "(unknown error)";
    }
    // On Win32 we'll use a message box. On !Win32, just print to stderr and abort()
#if defined(_WIN32)
//    MessageBox(0, errorString.c_str(), title.c_str(), MB_OK | MB_ICONEXCLAMATION);
#else
    fprintf(stderr, "%s : %s\n", title.c_str(), errorString.c_str());
#endif
    abort();
}

void gl_misc::non_fatal_error(std::string errorString, std::string title)
{
    if(title.empty())
    {
        title = "GL-Tutorial - Error";
    }
    if(errorString.empty())
    {
        errorString = "(unknown error)";
    }
    // On Win32 we'll use a message box. On !Win32, just print to stderr and abort()
#if defined(_WIN32)
//    MessageBox(0, errorString.c_str(), title.c_str(), MB_OK | MB_ICONEXCLAMATION);
#else
    fprintf(stderr, "%s : %s\n", title.c_str(), errorString.c_str());
#endif
}

bool gl_misc::checkGLError(const char* file, int line)
{
    bool wasError = false;

    for(GLenum glErr = glGetError(); glErr != GL_NO_ERROR; glErr = glGetError())
    {
        wasError = true;
        std::cout << "OpenGL Error" << std::endl;
    }
    return wasError;
}

// startupGLDiagnostics
void gl_misc::startupGLDiagnostics()
{
    // print diagnostic information
    printf("GL  VENDOR: %s\n", glGetString(GL_VENDOR));
    printf("   VERSION: %s\n", glGetString(GL_VERSION));
    printf("  RENDERER: %s\n", glGetString(GL_RENDERER));

    // test if we've got GL 3.0
    if(!GLEW_VERSION_3_0)
    {
        gl_misc::fatal_error("OpenGL 3.0 not supported.\n",
                    "Please update your drivers and/or buy a better graphics card.");
    }
}

namespace
{
#if defined(_WIN32)
#define CALLBACK_ CALLBACK
#else // !win32
#define CALLBACK_
#endif // ~ platform

    ///////////////////////////////////////////////////////////////////////////
    /// Callback function. This function is called by GL
    /// whenever it generates a debug message.
    ///////////////////////////////////////////////////////////////////////////
    GLvoid CALLBACK_ handle_debug_message_(GLenum aSource,
                                           GLenum aType,
                                           GLuint aId,
                                           GLenum aSeverity,
                                           GLsizei /*aLength*/,
                                           GLchar const* aMessage,
                                           GLvoid* /*aUser*/)
    {
        // source string
        const char* srcStr;
        switch(aSource)
        {
            case GL_DEBUG_SOURCE_API:
                srcStr = "API";
                break;
            case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
                srcStr = "WINDOW_SYSTEM";
                break;
            case GL_DEBUG_SOURCE_SHADER_COMPILER:
                srcStr = "SHADER_COMPILER";
                break;
            case GL_DEBUG_SOURCE_THIRD_PARTY:
                srcStr = "THIRD_PARTY";
                break;
            case GL_DEBUG_SOURCE_APPLICATION:
                srcStr = "APPLICATION";
                break;
            case GL_DEBUG_SOURCE_OTHER:
                srcStr = "OTHER";
                break;
            default:
                srcStr = "UNKNOWN";
                break;
        }

        // type
        const char* typeStr;
        switch(aType)
        {
            case GL_DEBUG_TYPE_ERROR:
                typeStr = "ERROR";
                break;
            case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
                typeStr = "DEPRECATED_BEHAVIOR";
                break;
            case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
                typeStr = "UNDEFINED_BEHAVIOR";
                break;
            case GL_DEBUG_TYPE_PORTABILITY:
                typeStr = "PORTABILITY";
                break;
            case GL_DEBUG_TYPE_PERFORMANCE:
                typeStr = "PERFORMANCE";
                break;
            case GL_DEBUG_TYPE_OTHER:
                typeStr = "OTHER";
                break;
            default:
                typeStr = "UNKNOWN";
                break;
        }

        // severity
        const char* sevStr;
        switch(aSeverity)
        {
            case GL_DEBUG_SEVERITY_HIGH:
                sevStr = "HIGH";
                break;
            case GL_DEBUG_SEVERITY_MEDIUM:
                sevStr = "MEDIUM";
                break;
            case GL_DEBUG_SEVERITY_LOW:
                sevStr = "LOW";
                break;
            case GL_DEBUG_SEVERITY_NOTIFICATION:
                sevStr = "NOTIFICATION";
                break;
            default:
                sevStr = "UNKNOWN";
        }

        // output message, if not just notification
        if(aSeverity != GL_DEBUG_SEVERITY_NOTIFICATION)
        {
            std::stringstream szs;
            szs << "\n"
                << "--\n"
                << "-- GL DEBUG MESSAGE:\n"
                << "--   severity = '" << sevStr << "'\n"
                << "--   type     = '" << typeStr << "'\n"
                << "--   source   = '" << srcStr << "'\n"
                << "--   id       = " << std::hex << aId << "\n"
                << "-- message:\n"
                << aMessage << "\n"
                << "--\n"
                << "\n";

            fprintf(stderr, "%s", szs.str().c_str());
            fflush(stderr);
#if defined(_WIN32)
            OutputDebugStringA(szs.str().c_str());
#endif
        }

        // Additionally: if it's (really) bad -> break!
        if(aSeverity == GL_DEBUG_SEVERITY_HIGH)
        {
#if defined(_WIN32)
            if(IsDebuggerPresent())
                __debugbreak();
#else  // !win32
            raise(SIGTRAP);
#endif // ~ platform
        }
    }

    // cleanup macros
#undef CALLBACK_
} // namespace

void gl_misc::setupGLDebugMessages()
{
    // Check for GL errors prior to us.
    CHECK_GL_ERROR()

    // Make sure that we support this extension before attempting to do anything with it...
    if(!glewIsExtensionSupported("GL_ARB_debug_output"))
    {
        printf(" --- \n");
        printf(" --- \n");
        printf(" --- BIG WARNING: GL_debug_output not supported!\n");
        printf(" --- This is rather bad news.\n"); /* TODO: give link to doc explaining this */
        printf(" --- \n");
        printf(" --- \n");
        return;
    }

    // Enable debug messages. Set a callback handle, and then enable all messages to begin with.
    glDebugMessageCallback((GLDEBUGPROC)handle_debug_message_, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, true);
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_PERFORMANCE, GL_DONT_CARE, 0, nullptr, false);

    // Enable synchronous debug output; this causes the callback to be called
    // immediately on error, usually in the actual gl-function where the error
    // occurred. (So, your stack trace is actually useful).
    //
    // This comes at a performance cost, so for performance sensitive stuff,
    // it might not be a good idea to unconditionally enable this. For the
    // labs, we just enable it, however.
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

    // Debug output can be somewhat spammy, especially if all messages are
    // enabled. For now, disable the lowest level of messages, which mostly
    // contain performance-related information and other random notes.
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_LOW, 0, nullptr, false);

    // Now, check if enabling debug messages caused a GL error. If so, that
    // error might not be reported by the debug message mechanism (after all,
    // the error occurred while setting it up). Later errors should be reported
    // via the callbacks, though.
    CHECK_GL_ERROR();
}