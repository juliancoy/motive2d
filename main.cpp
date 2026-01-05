#include "motive2d.h"

int main(int argc, char **argv){

    CliOptions opts{};
    bool windowsSpecified = false;
    bool parsedInput = false;
    bool parsedRegion = false;
    bool parsedGrading = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i] ? argv[i] : "");
        if (arg.empty())
        {
            continue;
        }
        if (arg == "--video" && i + 1 < argc)
        {
            std::string nextArg(argv[i + 1] ? argv[i + 1] : "");
            if (!nextArg.empty() && nextArg[0] != '-')
            {
                opts.videoPath = std::filesystem::path(nextArg);
                ++i;
            }
            continue;
        }
        if (arg.rfind("--video=", 0) == 0)
        {
            opts.videoPath = std::filesystem::path(arg.substr(std::string("--video=").size()));
            continue;
        }
        if (arg == "--swapUV")
        {
            opts.swapUV = true;
            continue;
        }
        if (arg == "--noSwapUV")
        {
            opts.swapUV = false;
            continue;
        }
        if (arg == "--no-overlays")
        {
            opts.overlaysEnabled = false;
            continue;
        }
        if (arg == "--overlays")
        {
            opts.overlaysEnabled = true;
            continue;
        }
        if (arg == "--debugDecode")
        {
            opts.debugDecode = true;
            continue;
        }
        if (arg == "--input-only")
        {
            opts.inputOnly = true;
            continue;
        }
        if (arg == "--skip-blit")
        {
            opts.skipBlit = true;
            continue;
        }
        if (arg == "--no-scrubber")
        {
            opts.scrubberEnabled = false;
            continue;
        }
        if (arg == "--scrubber")
        {
            opts.scrubberEnabled = true;
            continue;
        }
        if (arg == "--single-frame" || arg == "--first-frame-only")
        {
            opts.singleFrame = true;
            continue;
        }
        if (arg.rfind("--output=", 0) == 0)
        {
            opts.outputImagePath = std::filesystem::path(arg.substr(std::string("--output=").size()));
            continue;
        }
        if (arg == "--no-subtitle-background")
        {
            opts.subtitleBackground = false;
            continue;
        }
        if (arg == "--debug")
        {
            opts.debugLogging = true;
            continue;
        }
        if (arg.rfind("--windows", 0) == 0)
        {
            std::string list;
            if (arg == "--windows" && i + 1 < argc)
            {
                std::string nextArg(argv[i + 1] ? argv[i + 1] : "");
                if (!nextArg.empty() && nextArg[0] != '-')
                {
                    list = nextArg;
                    ++i;
                }
            }
            else if (arg.rfind("--windows=", 0) == 0)
            {
                list = arg.substr(std::string("--windows=").size());
            }

            if (!list.empty())
            {
                windowsSpecified = true;
                parsedInput = false;
                parsedRegion = false;
                parsedGrading = false;
                std::stringstream ss(list);
                std::string token;
                while (std::getline(ss, token, ','))
                {
                    if (token == "none")
                    {
                        parsedInput = parsedRegion = parsedGrading = false;
                        continue;
                    }
                    if (token == "input")
                    {
                        parsedInput = true;
                        continue;
                    }
                    if (token == "region")
                    {
                        parsedRegion = true;
                        continue;
                    }
                    if (token == "grading")
                    {
                        parsedGrading = true;
                    }
                }
            }
        }
        else if (arg == "--pose")
        {
            opts.poseEnabled = true;
            if (i + 1 < argc)
            {
                std::string nextArg(argv[i + 1] ? argv[i + 1] : "");
                if (!nextArg.empty() && nextArg[0] != '-')
                {
                    opts.poseModelBase = std::filesystem::path(nextArg);
                    ++i;
                }
            }
        }
        else if (arg.rfind("--pose=", 0) == 0)
        {
            opts.poseEnabled = true;
            opts.poseModelBase = std::filesystem::path(arg.substr(std::string("--pose=").size()));
        }
        else if (arg[0] != '-')
        {
            opts.videoPath = std::filesystem::path(arg);
        }
    }

    if (windowsSpecified)
    {
        opts.showInput = parsedInput;
        opts.showRegion = parsedRegion;
        opts.showGrading = parsedGrading;
    }

    if (opts.inputOnly)
    {
        opts.showInput = true;
        opts.showRegion = false;
        opts.showGrading = false;
        opts.overlaysEnabled = false;
        // Don't skip blit - we still want to see the video in the input window
        opts.skipBlit = false;
    }

    Motive2D app(opts);
    app.run(argc, argv);
    ~app;
}
