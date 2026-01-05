#pragma once

#include <stdbool.h>

bool debugLoggingEnabled();
void setDebugLoggingEnabled(bool enabled);

bool renderDebugEnabled();
void setRenderDebugEnabled(bool enabled);

#define LOG_DEBUG(expr)                     \
    do                                      \
    {                                       \
        if (renderDebugEnabled())           \
        {                                   \
            expr;                           \
        }                                   \
    } while (0)
