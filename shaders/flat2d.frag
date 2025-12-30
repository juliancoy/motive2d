#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0) uniform ObjectUBO {
    mat4 model;
    uvec4 instanceData;
    uvec4 yuvParams;
} objectUBO;

layout(set = 1, binding = 1) uniform sampler2D texSampler;
layout(set = 1, binding = 2) uniform sampler2D texChromaSampler;

const float kLumaOffset = 16.0 / 255.0;
const float kLumaScaleLimited = 255.0 / 219.0;
const float kChromaScaleLimited = 255.0 / 224.0;

float convertLuma(float ySample, uint rangeFlag) {
    if (rangeFlag == 0u) {
        return clamp((ySample - kLumaOffset) * kLumaScaleLimited, 0.0, 1.0);
    }
    return ySample;
}

vec2 convertChroma(vec2 uvSample, uint rangeFlag) {
    if (rangeFlag == 0u) {
        return (uvSample - vec2(0.5)) * kChromaScaleLimited;
    }
    return uvSample - vec2(0.5);
}

vec3 yuvToRgb(float y, float u, float v, uint colorSpace) {
    vec3 rgb;
    if (colorSpace == 2u) {
        rgb.r = y + 1.4746 * v;
        rgb.g = y - 0.164553 * u - 0.571353 * v;
        rgb.b = y + 1.8814 * u;
    } else if (colorSpace == 1u) {
        rgb.r = y + 1.5748 * v;
        rgb.g = y - 0.187324 * u - 0.468124 * v;
        rgb.b = y + 1.8556 * u;
    } else {
        rgb.r = y + 1.402 * v;
        rgb.g = y - 0.344136 * u - 0.714136 * v;
        rgb.b = y + 1.772 * u;
    }
    return clamp(rgb, 0.0, 1.0);
}

void main() {
    vec4 sampledColor = texture(texSampler, fragTexCoord);
    const uint yuvFormat = objectUBO.instanceData.y;
    if (yuvFormat != 0u) {
        const uint colorSpace = objectUBO.instanceData.z;
        const uint colorRange = objectUBO.instanceData.w;
        float ySample = convertLuma(sampledColor.r, colorRange);
        vec2 uvSample = texture(texChromaSampler, fragTexCoord).rg;
        vec2 uv = convertChroma(uvSample, colorRange);
        vec3 rgb = yuvToRgb(ySample, uv.x, uv.y, colorSpace);
        sampledColor = vec4(rgb, 1.0);
    }

    outColor = sampledColor;
}
