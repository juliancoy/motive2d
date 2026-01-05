#include "decoder.h"

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

namespace
{
void printUsage()
{
    std::cout << "Usage: decode_benchmark [--swapUV|--noSwapUV] <video-file>\n";
    std::cout << "Options:\n";
    std::cout << "  --swapUV       Request UV swap when copying planar chroma buffers.\n";
    std::cout << "  --noSwapUV     Disable UV swapping (default behavior).\n";
    std::cout << "  --help         Show this message.\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printUsage();
        return 1;
    }

    std::filesystem::path videoPath;
    std::optional<bool> swapUv;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i] ? argv[i] : "");
        if (arg.empty())
        {
            continue;
        }
        if (arg == "--swapUV")
        {
            swapUv = true;
            continue;
        }
        if (arg == "--noSwapUV")
        {
            swapUv = false;
            continue;
        }
        if (arg == "--help" || arg == "-h")
        {
            printUsage();
            return 0;
        }
        if (videoPath.empty())
        {
            videoPath = std::filesystem::path(arg);
            continue;
        }
        std::cerr << "Unrecognized argument: " << arg << "\n";
        printUsage();
        return 1;
    }

    if (videoPath.empty())
    {
        std::cerr << "No input video specified.\n";
        printUsage();
        return 1;
    }

    constexpr double kDecodeDurationSeconds = 10.0;
    return runDecodeOnlyBenchmark(videoPath, swapUv, kDecodeDurationSeconds);
}
