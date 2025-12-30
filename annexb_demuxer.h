#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>
#include <string>

// Minimal Annex-B elementary stream reader (H.264/H.265).
// No container parsing; expects raw .h264/.h265 streams.
class AnnexBDemuxer
{
public:
    explicit AnnexBDemuxer(const std::filesystem::path& path);
    ~AnnexBDemuxer();
    bool valid() const { return mapped; }

    // Returns next NALU range [data,data+size). size==0 on EOF.
    bool nextNalu(const uint8_t*& data, size_t& size, bool& isIdr);

    void rewind();

private:
    bool mapFile(const std::filesystem::path& path);
    const uint8_t* findStart(const uint8_t* p, const uint8_t* end);

    std::vector<uint8_t> buffer; // unused if mmap
    const uint8_t* base = nullptr; // start of mmap region
    const uint8_t* ptr = nullptr;
    const uint8_t* end = nullptr;
    bool mapped = false;
    size_t mappedSize = 0;
};
