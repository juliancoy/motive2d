#include "annexb_demuxer.h"

#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

AnnexBDemuxer::AnnexBDemuxer(const std::filesystem::path& path)
    : base(nullptr), ptr(nullptr), end(nullptr), mapped(false), mappedSize(0)
{
    mapped = mapFile(path);
}

AnnexBDemuxer::~AnnexBDemuxer()
{
    if (base && mappedSize > 0)
    {
        munmap(const_cast<uint8_t*>(base), mappedSize);
    }
}

bool AnnexBDemuxer::mapFile(const std::filesystem::path& path)
{
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1)
    {
        std::cerr << "[Demux] Failed to open " << path << "\n";
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) == -1)
    {
        std::cerr << "[Demux] Failed to stat " << path << "\n";
        close(fd);
        return false;
    }
    size_t fileSize = st.st_size;
    if (fileSize == 0)
    {
        std::cerr << "[Demux] File is empty\n";
        close(fd);
        return false;
    }
    void* mappedData = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mappedData == MAP_FAILED)
    {
        std::cerr << "[Demux] mmap failed for " << path << "\n";
        return false;
    }
    std::cout << "[Demux] Mapped " << fileSize << " bytes via mmap.\n";
    base = static_cast<const uint8_t*>(mappedData);
    ptr = base;
    end = base + fileSize;
    mappedSize = fileSize;
    mapped = true;
    std::cout << "[Demux] mapFile returning true.\n";
    return true;
}

const uint8_t* AnnexBDemuxer::findStart(const uint8_t* p, const uint8_t* e)
{
    while (p + 3 < e)
    {
        if (p[0] == 0 && p[1] == 0 && p[2] == 1)
            return p;
        if (p + 4 < e && p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 1)
            return p;
        ++p;
    }
    return e;
}

bool AnnexBDemuxer::nextNalu(const uint8_t*& data, size_t& size, bool& isIdr)
{
    if (!mapped || ptr >= end)
    {
        size = 0;
        return false;
    }
    const uint8_t* start = findStart(ptr, end);
    if (start == end)
    {
        size = 0;
        return false;
    }
    const uint8_t* nalStart = start + 3 + (start[2] == 0);
    const uint8_t* next = findStart(nalStart, end);
    data = nalStart;
    size = static_cast<size_t>(next - nalStart);
    ptr = next;

    // H.264: NAL type in lower 5 bits; H.265: lower 6 bits >>1
    uint8_t header = nalStart[0];
    uint8_t nalType = (header & 0x1F);
    if ((header & 0x7E) == 0) // likely H.264, leave nalType as-is
    {
        isIdr = (nalType == 5);
    }
    else
    {
        // H.265
        nalType = (header >> 1) & 0x3F;
        isIdr = (nalType >= 16 && nalType <= 21);
    }
    return true;
}

void AnnexBDemuxer::rewind()
{
    ptr = base;
}
