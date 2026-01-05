#include "debug_logging.h"

#include <cstdlib>

static bool gDebugLoggingEnabled = false;
static bool gRenderDebugEnabled = false;

bool debugLoggingEnabled()
{
    return gDebugLoggingEnabled;
}

void setDebugLoggingEnabled(bool enabled)
{
    gDebugLoggingEnabled = enabled;
}

bool renderDebugEnabled()
{
    return gRenderDebugEnabled || gDebugLoggingEnabled;
}

void setRenderDebugEnabled(bool enabled)
{
    gRenderDebugEnabled = enabled || (::getenv("MOTIVE2D_DEBUG_RENDER") != nullptr);
}
