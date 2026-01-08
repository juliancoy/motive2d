// decoder_vulkan.cpp
#include "decoder_vulkan.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}

#include "engine2d.h"

// Interrupt callback forward declaration
static int interrupt_callback(void *opaque);

// Returns a readable description of an FFmpeg pixel format.
static std::string pixelFormatDescription(AVPixelFormat fmt)
{
    const char* name = av_get_pix_fmt_name(fmt);
    if (!name) name = "unknown";

    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(fmt);
    if (!desc) {
        return std::string(name);
    }

    // Components / bit depth
    int maxDepth = 0;
    for (int i = 0; i < desc->nb_components; ++i) {
        maxDepth = std::max(maxDepth, desc->comp[i].depth);
    }

    const int chromaW = desc->log2_chroma_w;
    const int chromaH = desc->log2_chroma_h;

    std::string out;
    out.reserve(128);
    out += name;
    out += " (";
    out += std::to_string(desc->nb_components);
    out += "c, depth=";
    out += std::to_string(maxDepth);
    out += ", chroma=2^(-";
    out += std::to_string(chromaW);
    out += ", -";
    out += std::to_string(chromaH);
    out += ")";

    // Flags (planar, RGB, alpha, etc.)
    out += ", flags=";
    bool anyFlag = false;

    auto addFlag = [&](const char* f) {
        if (anyFlag) out += "|";
        out += f;
        anyFlag = true;
    };

    if (desc->flags & AV_PIX_FMT_FLAG_PLANAR)     addFlag("planar");
    if (desc->flags & AV_PIX_FMT_FLAG_RGB)        addFlag("rgb");
    if (desc->flags & AV_PIX_FMT_FLAG_ALPHA)      addFlag("alpha");
    if (desc->flags & AV_PIX_FMT_FLAG_PAL)        addFlag("pal");
    if (desc->flags & AV_PIX_FMT_FLAG_BITSTREAM)  addFlag("bitstream");
    if (desc->flags & AV_PIX_FMT_FLAG_HWACCEL)    addFlag("hwaccel");
    if (desc->flags & AV_PIX_FMT_FLAG_BAYER)      addFlag("bayer");
    if (desc->flags & AV_PIX_FMT_FLAG_FLOAT)      addFlag("float");

    if (!anyFlag) out += "none";

    return out;
}

// Local FFmpeg error-string helper
static std::string avErrStr(int err)
{
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(err, buf, sizeof(buf));
    return std::string(buf);
}

// ------------------------------
// DecodedFrame RAII
// ------------------------------
DecodedFrame::~DecodedFrame() { reset(); }

DecodedFrame::DecodedFrame(DecodedFrame&& other) noexcept {
    *this = std::move(other);
}

DecodedFrame& DecodedFrame::operator=(DecodedFrame&& other) noexcept {
    if (this == &other) return *this;
    reset();
    ptsSeconds = other.ptsSeconds;
    vk = other.vk;
    avFrame = other.avFrame;
    other.avFrame = nullptr;
    other.vk = VulkanSurface{};
    other.ptsSeconds = 0.0;
    return *this;
}

void DecodedFrame::reset() {
    if (avFrame) {
        av_frame_free(&avFrame);
        avFrame = nullptr;
    }
    ptsSeconds = 0.0;
    vk = VulkanSurface{};
}

// ------------------------------
// BoundedQueue impl
// ------------------------------
template <typename T, size_t Capacity>
bool BoundedQueue<T, Capacity>::push(T&& item) {
    std::unique_lock<std::mutex> lk(m_);
    cvNotFull_.wait(lk, [&]{ return stopped_ || count_ < Capacity; });
    if (stopped_) return false;

    buf_[tail_] = std::move(item);
    tail_ = (tail_ + 1) % Capacity;
    ++count_;

    lk.unlock();
    cvNotEmpty_.notify_one();
    return true;
}

template <typename T, size_t Capacity>
bool BoundedQueue<T, Capacity>::try_pop(T& out) {
    std::lock_guard<std::mutex> lk(m_);
    if (count_ == 0) return false;

    out = std::move(buf_[head_]);
    head_ = (head_ + 1) % Capacity;
    --count_;

    cvNotFull_.notify_one();
    return true;
}

template <typename T, size_t Capacity>
bool BoundedQueue<T, Capacity>::pop(T& out) {
    std::unique_lock<std::mutex> lk(m_);
    cvNotEmpty_.wait(lk, [&]{ return stopped_ || count_ > 0; });
    if (count_ == 0) return false;

    out = std::move(buf_[head_]);
    head_ = (head_ + 1) % Capacity;
    --count_;

    lk.unlock();
    cvNotFull_.notify_one();
    return true;
}

template <typename T, size_t Capacity>
void BoundedQueue<T, Capacity>::stop() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stopped_ = true;
    }
    cvNotEmpty_.notify_all();
    cvNotFull_.notify_all();
}

template <typename T, size_t Capacity>
void BoundedQueue<T, Capacity>::reset() {
    std::lock_guard<std::mutex> lk(m_);
    head_ = tail_ = count_ = 0;
    stopped_ = false;
}

template <typename T, size_t Capacity>
size_t BoundedQueue<T, Capacity>::size() const {
    std::lock_guard<std::mutex> lk(m_);
    return count_;
}

// Explicit instantiate the templates we use
template class BoundedQueue<DecodedFrame, DecoderVulkan::kBufferedFrames>;

// ------------------------------
// DecoderVulkan
// ------------------------------
DecoderVulkan::DecoderVulkan(const std::filesystem::path& videoPath, Engine2D* eng)
    : engine(eng)
{
    if (!openInputAndCodec(videoPath)) {
        valid = false;
        return;
    }

    if (engine) {
        sampler = createLinearClampSampler();
    }

    valid = true;
}

DecoderVulkan::~DecoderVulkan() {
    stopAsyncDecoding();
    destroyExternalVideoViews();

    if (engine && sampler != VK_NULL_HANDLE) {
        vkDestroySampler(engine->logicalDevice, sampler, nullptr);
        sampler = VK_NULL_HANDLE;
    }

    cleanupFFmpeg();
}

static AVPixelFormat chooseUsefulPixFmtForConfig(const AVCodecContext* c)
{
    if (!c) return AV_PIX_FMT_NONE;

    // With hwaccel, codecCtx->pix_fmt is usually the hw surface (e.g. AV_PIX_FMT_VULKAN),
    // and the *actual* subsampling/bit depth lives in sw_pix_fmt.
    if (c->pix_fmt == AV_PIX_FMT_VULKAN && c->sw_pix_fmt != AV_PIX_FMT_NONE)
        return c->sw_pix_fmt;

    // If pix_fmt is unknown but sw_pix_fmt is known, use sw_pix_fmt.
    if (c->pix_fmt == AV_PIX_FMT_NONE && c->sw_pix_fmt != AV_PIX_FMT_NONE)
        return c->sw_pix_fmt;

    return c->pix_fmt;
}

bool DecoderVulkan::openInputAndCodec(const std::filesystem::path& videoPath)
{
    if (avformat_open_input(&formatCtx, videoPath.string().c_str(), nullptr, nullptr) < 0) {
        throw std::runtime_error("[DecoderVulkan] Failed to open input: " + videoPath.string());
    }

    formatCtx->interrupt_callback.callback = interrupt_callback;
    formatCtx->interrupt_callback.opaque = this;

    if (avformat_find_stream_info(formatCtx, nullptr) < 0) {
        throw std::runtime_error("[DecoderVulkan] Failed to find stream info.");
    }

    videoStreamIndex = av_find_best_stream(formatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (videoStreamIndex < 0) {
        throw std::runtime_error("[DecoderVulkan] No video stream found.");
    }

    AVStream* videoStream = formatCtx->streams[videoStreamIndex];
    streamTimeBase = videoStream->time_base;

    const AVCodec* codec = avcodec_find_decoder(videoStream->codecpar->codec_id);
    if (!codec) {
        throw std::runtime_error("[DecoderVulkan] Decoder not found.");
    }

    codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx) {
        throw std::runtime_error("[DecoderVulkan] Failed to allocate codec context.");
    }

    if (avcodec_parameters_to_context(codecCtx, videoStream->codecpar) < 0) {
        throw std::runtime_error("[DecoderVulkan] Failed to copy codec parameters.");
    }

    // Initialize Vulkan hw device context (requires Engine2D)
    if (!initFFmpegVulkanDevice()) {
        return false;
    }

    // Manually create a hw_frames_ctx to override the image format selected by FFmpeg.
    // The default format (VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) used for NV12 is not
    // supported by the user's driver for video decoding. We will force a 2-plane
    // format using VK_FORMAT_R8_UNORM and VK_FORMAT_R8G8_UNORM, which is a common fallback.
    AVBufferRef* hw_frames_ref = av_hwframe_ctx_alloc(codecCtx->hw_device_ctx);
    if (!hw_frames_ref) {
        throw std::runtime_error("[DecoderVulkan] av_hwframe_ctx_alloc failed.");
    }
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)(hw_frames_ref->data);
    AVVulkanFramesContext* vk_frames_ctx = (AVVulkanFramesContext*)frames_ctx->hwctx;

    frames_ctx->format = AV_PIX_FMT_VULKAN;
    frames_ctx->sw_format = AV_PIX_FMT_NV12; // Assume NV12, which matches the failing format
    frames_ctx->width = codecCtx->width;
    frames_ctx->height = codecCtx->height;

    // Set the Vulkan formats for the two planes of NV12
    vk_frames_ctx->format[0] = VK_FORMAT_R8_UNORM;   // Y plane
    vk_frames_ctx->format[1] = VK_FORMAT_R8G8_UNORM; // Interleaved UV plane

    // Set the image usage flags required for video decoding
    vk_frames_ctx->usage = static_cast<VkImageUsageFlagBits>(
                           VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
                           VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
                           VK_IMAGE_USAGE_SAMPLED_BIT |
                           VK_IMAGE_USAGE_STORAGE_BIT |
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    // Set image creation flags, including PROFILE_INDEPENDENT to resolve the VUID-VkImageCreateInfo-usage-04815 error.
    vk_frames_ctx->img_flags = VK_IMAGE_CREATE_VIDEO_PROFILE_INDEPENDENT_BIT_KHR |
                               VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                               VK_IMAGE_CREATE_ALIAS_BIT |
                               VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;

    int ret = av_hwframe_ctx_init(hw_frames_ref);
    if (ret < 0) {
        av_buffer_unref(&hw_frames_ref);
        throw std::runtime_error("[DecoderVulkan] av_hwframe_ctx_init failed: " + avErrStr(ret));
    }
    codecCtx->hw_frames_ctx = av_buffer_ref(hw_frames_ref);
    av_buffer_unref(&hw_frames_ref);

    // Allow FFmpeg internal threading
    unsigned int hwThreads = std::thread::hardware_concurrency();
    codecCtx->thread_count = hwThreads > 0 ? static_cast<int>(hwThreads) : 0;
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        throw std::runtime_error("[DecoderVulkan] Failed to open codec.");
    }

    width  = static_cast<uint32_t>(codecCtx->width);
    height = static_cast<uint32_t>(codecCtx->height);

    frame = av_frame_alloc();
    packet = av_packet_alloc();
    if (!frame || !packet) {
        throw std::runtime_error("[DecoderVulkan] Failed to allocate AVFrame/AVPacket.");
    }

    // FPS + duration
    AVRational fr = av_guess_frame_rate(formatCtx, videoStream, nullptr);
    fps = (fr.num && fr.den) ? av_q2d(fr) : 30.0;
    fps = fps > 0.0 ? fps : 30.0;

    durationSeconds = 0.0;
    if (formatCtx->duration > 0) {
        durationSeconds = static_cast<double>(formatCtx->duration) / static_cast<double>(AV_TIME_BASE);
    }

    // Pixel format config:
    // Use sw_pix_fmt when decode output is AV_PIX_FMT_VULKAN.
    const AVPixelFormat cfgFmt = chooseUsefulPixFmtForConfig(codecCtx);
    if (!configureFormatForPixelFormat(cfgFmt)) {
        throw std::runtime_error("[DecoderVulkan] Unsupported pixel format: " +
                                 pixelFormatDescription(cfgFmt));
    }

    // Helpful logging: shows both hw pix_fmt and sw_pix_fmt.
    std::cerr << "[DecoderVulkan] codecCtx->pix_fmt=" << pixelFormatDescription(codecCtx->pix_fmt)
              << " sw_pix_fmt=" << pixelFormatDescription(codecCtx->sw_pix_fmt)
              << " (config fmt=" << pixelFormatDescription(cfgFmt) << ")\n";

    return true;
}

bool DecoderVulkan::initFFmpegVulkanDevice()
{
    if (!engine) {
        hardwareInitFailureReason = "Engine2D is null (no Vulkan device available)";
        return false;
    }

    hwDeviceCtx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
    if (!hwDeviceCtx) {
        hardwareInitFailureReason = "Failed to allocate Vulkan hwdevice context";
        return false;
    }

    AVHWDeviceContext* deviceCtx = reinterpret_cast<AVHWDeviceContext*>(hwDeviceCtx->data);
    AVVulkanDeviceContext* vkctx = static_cast<AVVulkanDeviceContext*>(deviceCtx->hwctx);
    std::memset(vkctx, 0, sizeof(*vkctx));

    vkctx->inst = engine->instance;
    vkctx->phys_dev = engine->physicalDevice;
    vkctx->act_dev = engine->logicalDevice;
    vkctx->get_proc_addr = vkGetInstanceProcAddr;

    // Deprecated fields (kept for compatibility with some FFmpeg versions/builds)
    vkctx->queue_family_index = engine->graphicsQueueFamilyIndex;
    vkctx->nb_graphics_queues = 1;

    vkctx->queue_family_tx_index = engine->graphicsQueueFamilyIndex;
    vkctx->nb_tx_queues = 1;

    vkctx->queue_family_comp_index = engine->graphicsQueueFamilyIndex;
    vkctx->nb_comp_queues = 1;

    vkctx->queue_family_decode_index = engine->videoQueueFamilyIndex;
    vkctx->nb_decode_queues = 1;

    vkctx->queue_family_encode_index = -1;
    vkctx->nb_encode_queues = 0;

    // New API queue-family list
    int q = 0;
    vkctx->qf[q].idx = engine->graphicsQueueFamilyIndex;
    vkctx->qf[q].num = 1;
    vkctx->qf[q].flags = static_cast<VkQueueFlagBits>(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT);
    vkctx->qf[q].video_caps = static_cast<VkVideoCodecOperationFlagBitsKHR>(0);
    q++;

    vkctx->qf[q].idx = engine->videoQueueFamilyIndex;
    vkctx->qf[q].num = 1;
    vkctx->qf[q].flags = static_cast<VkQueueFlagBits>(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_VIDEO_DECODE_BIT_KHR);
    vkctx->qf[q].video_caps = static_cast<VkVideoCodecOperationFlagBitsKHR>(
        VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR |
        VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR);
    q++;

    // Some FFmpeg builds expect “extra” entries even if they duplicate graphics
    vkctx->qf[q].idx = engine->graphicsQueueFamilyIndex;
    vkctx->qf[q].num = 1;
    vkctx->qf[q].flags = static_cast<VkQueueFlagBits>(VK_QUEUE_TRANSFER_BIT);
    vkctx->qf[q].video_caps = static_cast<VkVideoCodecOperationFlagBitsKHR>(0);
    q++;

    vkctx->qf[q].idx = engine->graphicsQueueFamilyIndex;
    vkctx->qf[q].num = 1;
    vkctx->qf[q].flags = static_cast<VkQueueFlagBits>(VK_QUEUE_COMPUTE_BIT);
    vkctx->qf[q].video_caps = static_cast<VkVideoCodecOperationFlagBitsKHR>(0);
    q++;

    vkctx->nb_qf = q;

    // Let FFmpeg manage queue locking internally
    vkctx->lock_queue = nullptr;
    vkctx->unlock_queue = nullptr;

        // Extensions.
        // NOTE: Keeping Vulkan Video extensions enabled is what triggers FFmpeg/driver probing.
        // If you want a "safe fallback" mode, you can gate these behind a runtime flag and
        // disable them when the source stream is known to be unsupported (e.g. H.264 4:2:2 10-bit).
        static const char* instanceExts[] = { "VK_KHR_surface", "VK_KHR_xcb_surface" };
        static const char* deviceExts[] = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,

            // Try disabling H.265 video decode extension to see if it helps with format issues
            VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
            VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
            VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME,
            VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME,

            // optional / nice-to-have depending on your driver/build:
            VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
            VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
            VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
            VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME,
            VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
        };

    vkctx->enabled_inst_extensions = instanceExts;
    vkctx->nb_enabled_inst_extensions = 2;

    vkctx->enabled_dev_extensions = deviceExts;
    vkctx->nb_enabled_dev_extensions = static_cast<int>(sizeof(deviceExts) / sizeof(deviceExts[0]));

    // Features (FIX: use vkGetPhysicalDeviceFeatures2, not vkGetPhysicalDeviceFeatures into features2.features)
    VkPhysicalDeviceFeatures2 features2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    vkGetPhysicalDeviceFeatures2(engine->physicalDevice, &features2);
    vkctx->device_features = features2;

    const int ret = av_hwdevice_ctx_init(hwDeviceCtx);
    if (ret < 0) {
        hardwareInitFailureReason = "av_hwdevice_ctx_init(Vulkan) failed: " + avErrStr(ret);
        av_buffer_unref(&hwDeviceCtx);
        hwDeviceCtx = nullptr;
        return false;
    }

    codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx);
    return true;
}

void DecoderVulkan::cleanupFFmpeg()
{
    if (packet) av_packet_free(&packet);
    if (frame) av_frame_free(&frame);

    if (hwDeviceCtx) {
        av_buffer_unref(&hwDeviceCtx);
        hwDeviceCtx = nullptr;
    }

    if (codecCtx) avcodec_free_context(&codecCtx);
    if (formatCtx) avformat_close_input(&formatCtx);
}

bool DecoderVulkan::configureFormatForPixelFormat(AVPixelFormat pix_fmt)
{
    sourcePixelFormat = pix_fmt;

    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(pix_fmt);
    if (!desc) return false;

    chromaDivX = 1u << desc->log2_chroma_w;
    chromaDivY = 1u << desc->log2_chroma_h;

    chromaWidth  = std::max<uint32_t>(1u, (width  + chromaDivX - 1) / chromaDivX);
    chromaHeight = std::max<uint32_t>(1u, (height + chromaDivY - 1) / chromaDivY);

    swapChromaUV = false;
    if (pix_fmt == AV_PIX_FMT_NV12) {
        swapChromaUV = false;
    }
    else if (pix_fmt == AV_PIX_FMT_NV21) {
        swapChromaUV = true;
    }
    else {
        // Accept common planar subsampling shapes if you later add planar support.
        const bool is420 = (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 1);
        const bool is422 = (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 0);
        const bool is444 = (desc->log2_chroma_w == 0 && desc->log2_chroma_h == 0);
        if (! (is420 || is422 || is444)) {
            throw std::runtime_error(
                "[DecoderVulkan] Unsupported pixel format (only NV12/NV21/4:2:0, 4:2:2, 4:4:4 supported): " +
                pixelFormatDescription(pix_fmt));
        }
    }

    // FIX: compute max component depth (not just comp[0])
    int maxDepth = 0;
    for (int i = 0; i < desc->nb_components; ++i) {
        maxDepth = std::max(maxDepth, desc->comp[i].depth);
    }
    bitDepth = maxDepth;
    bytesPerComponent = (bitDepth > 8) ? 2 : 1;

    return true;
}

// ------------------------------
// Async decode (producer)
// ------------------------------
bool DecoderVulkan::startAsyncDecoding()
{
    if (asyncDecoding) return true;

    stopRequested.store(false);
    threadRunning.store(true);
    finished.store(false);
    draining.store(false);

    decodedQ.reset();
    candidate.reset();

    asyncDecoding = true;
    decodeThread = std::thread(&DecoderVulkan::asyncDecodeLoop, this);
    return true;
}

void DecoderVulkan::stopAsyncDecoding()
{
    if (!asyncDecoding) return;

    stopRequested.store(true);
    decodedQ.stop();

    if (decodeThread.joinable())
        decodeThread.join();

    asyncDecoding = false;
    threadRunning.store(false);

    decodedQ.reset();
    candidate.reset();
}

void DecoderVulkan::asyncDecodeLoop()
{
    try {
        while (!stopRequested.load()) {
            DecodedFrame f;
            if (!decodeNextFrame(f)) break;

            // seek-drop logic
            const int64_t target = seekTargetMicroseconds.load();
            if (target >= 0) {
                const int64_t micros = static_cast<int64_t>(f.ptsSeconds * 1'000'000.0);
                if (micros < target) {
                    continue; // drop
                }
                seekTargetMicroseconds.store(-1);
            }

            if (!f.vk.validate()) {
                throw std::runtime_error("[DecoderVulkan] decoded frame missing/invalid Vulkan surface");
            }

            if (!decodedQ.push(std::move(f))) {
                break; // stopped
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[DecoderVulkan] asyncDecodeLoop exception: " << e.what() << std::endl;
    }

    threadRunning.store(false);
    decodedQ.stop();
}

// ------------------------------
// Decode one frame (keeps AVFrame alive in DecodedFrame)
// ------------------------------
bool DecoderVulkan::decodeNextFrame(DecodedFrame& out)
{
    if (finished.load()) return false;

    while (true) {
        if (!draining.load()) {
            const int rr = av_read_frame(formatCtx, packet);
            if (rr >= 0) {
                if (packet->stream_index == videoStreamIndex) {
                    const int sp = avcodec_send_packet(codecCtx, packet);
                    if (sp < 0) {
                        av_packet_unref(packet);
                        throw std::runtime_error("[DecoderVulkan] avcodec_send_packet failed: " + avErrStr(sp));
                    }
                }
                av_packet_unref(packet);
            } else {
                av_packet_unref(packet);
                draining.store(true);
                avcodec_send_packet(codecCtx, nullptr);
            }
        }

        const int rf = avcodec_receive_frame(codecCtx, frame);

        if (rf == 0) {
            const AVPixelFormat frameFmt = static_cast<AVPixelFormat>(frame->format);

            // If dimensions / format change midstream, update config
            if (width != static_cast<uint32_t>(frame->width) ||
                height != static_cast<uint32_t>(frame->height))
            {
                width  = static_cast<uint32_t>(frame->width);
                height = static_cast<uint32_t>(frame->height);
            }

            // When hwaccel is used, frameFmt is AV_PIX_FMT_VULKAN; configureFormat must be based on sw_format.
            // If the underlying sw format changes (rare, but can happen), update config.
            const AVPixelFormat cfgFmt = (frameFmt == AV_PIX_FMT_VULKAN)
                ? codecCtx->sw_pix_fmt
                : frameFmt;

            if (cfgFmt != sourcePixelFormat) {
                if (!configureFormatForPixelFormat(cfgFmt)) {
                    av_frame_unref(frame);
                    throw std::runtime_error("[DecoderVulkan] Unsupported pixel format during decode: " +
                                             pixelFormatDescription(cfgFmt));
                }
            }

            out.reset();
            out.avFrame = av_frame_clone(frame);
            if (!out.avFrame) {
                av_frame_unref(frame);
                throw std::runtime_error("[DecoderVulkan] av_frame_clone failed");
            }

            // Expect Vulkan output for zero-copy
            if (out.avFrame->format != AV_PIX_FMT_VULKAN) {
                const std::string got = pixelFormatDescription(static_cast<AVPixelFormat>(out.avFrame->format));
                av_frame_unref(frame);
                throw std::runtime_error(
                    "[DecoderVulkan] Decoder did not output AV_PIX_FMT_VULKAN (got " + got +
                    "). If this is expected for some sources, add a software->Vulkan upload path or disable GPU decode.");
            }

            const AVVkFrame* vkf = reinterpret_cast<const AVVkFrame*>(out.avFrame->data[0]);
            if (!vkf) {
                av_frame_unref(frame);
                throw std::runtime_error("[DecoderVulkan] AVVkFrame is null");
            }

            out.vk.valid = true;
            out.vk.width  = static_cast<uint32_t>(out.avFrame->width);
            out.vk.height = static_cast<uint32_t>(out.avFrame->height);

            int nb_images = 0;
            while (nb_images < AV_NUM_DATA_POINTERS && vkf->img[nb_images]) nb_images++;
            out.vk.planes = std::min<uint32_t>(static_cast<uint32_t>(std::max(nb_images, 0)), 3u);

            const VkFormat* vkFormats = av_vkfmt_from_pixfmt(codecCtx->sw_pix_fmt);

            for (uint32_t i = 0; i < out.vk.planes; ++i) {
                out.vk.images[i] = vkf->img[i];
                out.vk.layouts[i] = vkf->layout[i];
                out.vk.semaphores[i] = vkf->sem[i];
                out.vk.semaphoreValues[i] = vkf->sem_value[i];
                out.vk.queueFamily[i] = vkf->queue_family[i];
                out.vk.planeFormats[i] = vkFormats ? vkFormats[i] : VK_FORMAT_UNDEFINED;
            }

            // Compute PTS seconds
            double ptsSeconds = fallbackPtsSeconds;
            const int64_t bestTs = out.avFrame->best_effort_timestamp;
            if (bestTs != AV_NOPTS_VALUE) {
                const double tb = (streamTimeBase.den != 0)
                    ? (static_cast<double>(streamTimeBase.num) / static_cast<double>(streamTimeBase.den))
                    : 0.0;
                ptsSeconds = tb * static_cast<double>(bestTs);
            } else {
                const double frameDuration = fps > 0.0 ? (1.0 / fps) : (1.0 / 30.0);
                ptsSeconds = framesDecoded > 0 ? (fallbackPtsSeconds + frameDuration) : 0.0;
            }

            fallbackPtsSeconds = ptsSeconds;
            framesDecoded++;
            out.ptsSeconds = ptsSeconds;

            av_frame_unref(frame);
            return true;
        }

        if (rf == AVERROR(EAGAIN)) {
            continue;
        }

        if (rf == AVERROR_EOF) {
            finished.store(true);
            return false;
        }

        std::cerr << "[DecoderVulkan] avcodec_receive_frame error: " << avErrStr(rf)
                  << " (" << rf << ")\n";
        return false;
    }
}

// ------------------------------
// Vulkan helpers
// ------------------------------
VkSampler DecoderVulkan::createLinearClampSampler()
{
    if (!engine) return VK_NULL_HANDLE;

    VkSampler s = VK_NULL_HANDLE;
    VkSamplerCreateInfo ci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    ci.magFilter = VK_FILTER_LINEAR;
    ci.minFilter = VK_FILTER_LINEAR;
    ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.unnormalizedCoordinates = VK_FALSE;

    if (vkCreateSampler(engine->logicalDevice, &ci, nullptr, &s) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateSampler failed");
    }
    return s;
}

void DecoderVulkan::destroyExternalVideoViews()
{
    if (!engine) return;

    if (externalLumaView != VK_NULL_HANDLE) {
        vkDestroyImageView(engine->logicalDevice, externalLumaView, nullptr);
        externalLumaView = VK_NULL_HANDLE;
    }
    if (externalChromaView != VK_NULL_HANDLE) {
        vkDestroyImageView(engine->logicalDevice, externalChromaView, nullptr);
        externalChromaView = VK_NULL_HANDLE;
    }
    usingExternal = false;
}

VkImageView DecoderVulkan::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect)
{
    if (!engine) return VK_NULL_HANDLE;

    VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.baseMipLevel = 0;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.baseArrayLayer = 0;
    vi.subresourceRange.layerCount = 1;

    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(engine->logicalDevice, &vi, nullptr, &view) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return view;
}

bool DecoderVulkan::waitForVulkanFrameReady(const VulkanSurface& s)
{
    if (!engine) return false;
    if (!s.validate()) return false;

    std::vector<VkSemaphore> sems;
    std::vector<uint64_t> vals;
    sems.reserve(s.planes);
    vals.reserve(s.planes);

    for (uint32_t i = 0; i < s.planes; ++i) {
        if (s.semaphores[i] != VK_NULL_HANDLE) {
            sems.push_back(s.semaphores[i]);
            vals.push_back(s.semaphoreValues[i]);
        }
    }

    if (sems.empty()) return true;

    VkSemaphoreWaitInfo wi{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
    wi.semaphoreCount = static_cast<uint32_t>(sems.size());
    wi.pSemaphores = sems.data();
    wi.pValues = vals.data();

    return vkWaitSemaphores(engine->logicalDevice, &wi, UINT64_MAX) == VK_SUCCESS;
}

bool DecoderVulkan::createExternalViewsFromSurface(const VulkanSurface& s)
{
    if (!engine) return false;
    if (!s.validate()) return false;

    destroyExternalVideoViews();

    // For 3-plane YUV420P: plane 0 = Y, plane 1 = U, plane 2 = V
    // For 2-plane NV12: plane 0 = Y, plane 1 = UV interleaved
    VkFormat f0 = s.planeFormats[0] != VK_FORMAT_UNDEFINED ? s.planeFormats[0] : VK_FORMAT_R8_UNORM;
    externalLumaView = createImageView(s.images[0], f0, VK_IMAGE_ASPECT_COLOR_BIT);

    if (s.planes > 1) {
        // For 2-plane NV12, chroma is interleaved UV (R8G8)
        // For 3-plane YUV420P, we might want to create separate U and V views,
        // but the current pipeline expects only 2 views (luma + chroma).
        // For now, create chroma view from plane 1 (U plane for 3-plane).
        VkFormat f1 = s.planeFormats[1] != VK_FORMAT_UNDEFINED ? s.planeFormats[1] : VK_FORMAT_R8_UNORM;
        externalChromaView = createImageView(s.images[1], f1, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // TODO: For 3-plane YUV420P, we might need to handle plane 2 (V) separately
    // or modify the compute shader to sample from 3 separate textures.

    usingExternal =
        (externalLumaView != VK_NULL_HANDLE) &&
        (s.planes == 1 || externalChromaView != VK_NULL_HANDLE);

    return usingExternal;
}

// ------------------------------
// Playback timing (consumer owns FPS)
// ------------------------------
void DecoderVulkan::resetPlaybackClock()
{
    clockInitialized = false;
    firstPtsSeconds = 0.0;
    lastFramePtsSeconds = 0.0;
    lastDisplayedSeconds = 0.0;
    candidate.reset();
}

void DecoderVulkan::updatePlaybackTimestamps(const DecodedFrame& frame, std::chrono::steady_clock::time_point now)
{
    lastFramePtsSeconds = frame.ptsSeconds;
    lastFrameRenderWall = now;
    lastDisplayedSeconds = std::max(0.0, frame.ptsSeconds - firstPtsSeconds);
}

bool DecoderVulkan::shouldDisplayNow(const DecodedFrame& frame, std::chrono::steady_clock::time_point now) const
{
    if (!clockInitialized) return true;

    const double ptsOffset = frame.ptsSeconds - firstPtsSeconds;
    const auto target = playbackStartWall + std::chrono::duration<double>(ptsOffset);

    return (now + std::chrono::milliseconds(1) >= target);
}

void DecoderVulkan::dropLateFrames(std::chrono::steady_clock::time_point now)
{
    if (!clockInitialized) return;

    while (true) {
        DecodedFrame tmp;
        if (!decodedQ.try_pop(tmp)) break;

        const double ptsOffset = tmp.ptsSeconds - firstPtsSeconds;
        const auto target = playbackStartWall + std::chrono::duration<double>(ptsOffset);

        if (now + std::chrono::milliseconds(1) >= target) {
            continue; // drop
        } else {
            candidate = std::move(tmp);
            break;
        }
    }
}

double DecoderVulkan::advancePlayback()
{
    if (!playing) return lastDisplayedSeconds;

    auto now = std::chrono::steady_clock::now();

    if (!candidate) {
        DecodedFrame f;
        if (!decodedQ.try_pop(f)) {
            return lastDisplayedSeconds;
        }
        candidate = std::move(f);
    }

    if (!clockInitialized) {
        clockInitialized = true;
        firstPtsSeconds = candidate->ptsSeconds;
        playbackStartWall = now;
    }

    {
        const double ptsOffset = candidate->ptsSeconds - firstPtsSeconds;
        const auto target = playbackStartWall + std::chrono::duration<double>(ptsOffset);
        if (now - target > std::chrono::milliseconds(50)) {
            dropLateFrames(now);
            if (!candidate) return lastDisplayedSeconds;
        }
    }

    if (!shouldDisplayNow(*candidate, now)) {
        return lastDisplayedSeconds;
    }

    if (engine && candidate->vk.validate()) {
        if (!waitForVulkanFrameReady(candidate->vk)) {
            std::cerr << "[DecoderVulkan] vkWaitSemaphores failed\n";
        } else {
            createExternalViewsFromSurface(candidate->vk);
        }
    }

    updatePlaybackTimestamps(*candidate, now);
    candidate.reset();
    return lastDisplayedSeconds;
}

// ------------------------------
// Seek
// ------------------------------
bool DecoderVulkan::seek(float timeSeconds)
{
    if (!formatCtx || !codecCtx) return false;

    timeSeconds = std::clamp(timeSeconds, 0.0f, static_cast<float>(durationSeconds));

    const bool wasAsync = asyncDecoding;
    if (wasAsync) stopAsyncDecoding();

    decodedQ.reset();
    seekTargetMicroseconds.store(static_cast<int64_t>(timeSeconds * 1'000'000.0));

    AVStream* st = formatCtx->streams[videoStreamIndex];
    const int64_t targetTs =
        av_rescale_q(static_cast<int64_t>(timeSeconds * AV_TIME_BASE),
                     AV_TIME_BASE_Q,
                     st->time_base);

    const int ret = avformat_seek_file(formatCtx,
                                       videoStreamIndex,
                                       std::numeric_limits<int64_t>::min(),
                                       targetTs,
                                       targetTs,
                                       AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        seekTargetMicroseconds.store(-1);
        if (wasAsync) startAsyncDecoding();
        throw std::runtime_error("[DecoderVulkan] seek failed: " + avErrStr(ret));
    }

    avcodec_flush_buffers(codecCtx);
    avformat_flush(formatCtx);

    finished.store(false);
    draining.store(false);
    framesDecoded = 0;
    fallbackPtsSeconds = timeSeconds;

    destroyExternalVideoViews();
    resetPlaybackClock();

    if (wasAsync) startAsyncDecoding();
    return true;
}

// ------------------------------
// Benchmark helper (decode-only)
// ------------------------------
int runDecodeOnlyBenchmark(const std::filesystem::path &videoPath, double benchmarkSeconds)
{
    const double kBenchmarkSeconds = benchmarkSeconds > 0.0 ? benchmarkSeconds : 5.0;

    DecoderVulkan d(videoPath, /*engine*/nullptr);
    if (!d.valid) {
        std::cerr << "[DecodeOnly] Decoder invalid: " << d.getHardwareInitFailureReason() << "\n";
        return 1;
    }

    DecodedFrame f;
    auto start = std::chrono::steady_clock::now();
    size_t frames = 0;
    double firstPts = -1.0;

    while (d.decodeNextFrame(f)) {
        frames++;
        if (firstPts < 0.0) firstPts = f.ptsSeconds;
        if (f.ptsSeconds - firstPts >= kBenchmarkSeconds) break;
        f.reset();
    }

    auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    const double outFps = seconds > 0.0 ? (double)frames / seconds : 0.0;

    std::cout << "[DecodeOnly] Decoded " << frames << " frames in " << seconds
              << "s -> " << outFps << " fps\n";
    return 0;
}

// ------------------------------
// Interrupt callback
// ------------------------------
static int interrupt_callback(void *opaque)
{
    auto* d = static_cast<DecoderVulkan*>(opaque);
    return (d && d->isStopRequested()) ? 1 : 0;
}
