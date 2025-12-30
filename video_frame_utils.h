#pragma once

#include <vector>

#include "video.h"

extern "C" {
#include <libavutil/pixfmt.h>
struct AVFrame;
}

namespace video {

bool configureFormatForPixelFormat(VideoDecoder& decoder, AVPixelFormat pixFormat);
void copyDecodedFrameToBuffer(const VideoDecoder& decoder, AVFrame* frame, std::vector<uint8_t>& buffer);

} // namespace video
