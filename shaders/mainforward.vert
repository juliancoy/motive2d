#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
} cameraUBO;

const uint MAX_INSTANCE_COUNT = 128;
layout(set = 1, binding = 0) uniform ObjectUBO {
    mat4 model;
    uvec4 instanceData;
    uvec4 yuvParams;
} objectUBO;

struct InstanceTransform {
    vec4 offset;
    vec4 rotation; // xyz = Euler angles (radians), w = cap flag
};

layout(std140, set = 1, binding = 3) uniform InstanceTransformBuffer {
    InstanceTransform transforms[MAX_INSTANCE_COUNT];
} instanceBuffer;
 
vec3 rotateByEuler(vec3 v, vec3 angles) {
    float sx = sin(angles.x);
    float cx = cos(angles.x);
    vec3 rotated = vec3(v.x,
                        v.y * cx - v.z * sx,
                        v.y * sx + v.z * cx);
    float sy = sin(angles.y);
    float cy = cos(angles.y);
    rotated = vec3(rotated.x * cy + rotated.z * sy,
                   rotated.y,
                   -rotated.x * sy + rotated.z * cy);
    float sz = sin(angles.z);
    float cz = cos(angles.z);
    rotated = vec3(rotated.x * cz - rotated.y * sz,
                   rotated.x * sz + rotated.y * cz,
                   rotated.z);
    return rotated;
}

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    const float CAP_SEGMENT_COUNT = 8.0;
    const float CAP_ANGLE_STEP = 6.28318530718 / CAP_SEGMENT_COUNT;

    uint instanceCount = max(objectUBO.instanceData.x, 1u);
    uint instanceIdx = min(gl_InstanceIndex, instanceCount - 1u);
    vec3 instanceOffset = vec3(0.0);
    vec3 instanceRotation = vec3(0.0);
    float capFlag = 0.0;
    if (objectUBO.instanceData.x > 0u) {
        InstanceTransform inst = instanceBuffer.transforms[instanceIdx];
        instanceOffset = inst.offset.xyz;
        instanceRotation = inst.rotation.xyz;
        capFlag = inst.rotation.w;
    }

    mat4 modelMatrix = objectUBO.model;
    mat3 normalMatrix = mat3(modelMatrix);
    vec4 worldPosition;
    vec3 worldNormal;
    bool isCap = abs(capFlag) > 0.5;
    if (isCap) {
        mat4 invModel = inverse(modelMatrix);
        vec3 localOffset = (invModel * vec4(instanceOffset, 1.0)).xyz;
        float radius = max(length(localOffset.xz), 1e-6);
        float centralAngle = atan(localOffset.z, localOffset.x);
        float halfStep = CAP_ANGLE_STEP * 0.5;
        float angleStart = centralAngle - halfStep;
        float angleEnd = centralAngle + halfStep;
        float u = clamp(inTexCoord.x, 0.0, 1.0);
        float v = clamp(1.0 - inTexCoord.y, 0.0, 1.0);
        vec3 edgeStart = vec3(cos(angleStart) * radius, localOffset.y, sin(angleStart) * radius);
        vec3 edgeEnd = vec3(cos(angleEnd) * radius, localOffset.y, sin(angleEnd) * radius);
        vec3 rimPoint = mix(edgeStart, edgeEnd, u);
        vec3 center = vec3(0.0, localOffset.y, 0.0);
        vec3 localPosition = mix(rimPoint, center, v);
        worldPosition = modelMatrix * vec4(localPosition, 1.0);
        vec3 capNormal = vec3(0.0, capFlag > 0.0 ? 1.0 : -1.0, 0.0);
        worldNormal = normalMatrix * capNormal;
    } else {
        vec3 rotatedPosition = rotateByEuler(inPosition, instanceRotation);
        vec3 rotatedNormal = rotateByEuler(inNormal, instanceRotation);
        worldPosition = modelMatrix * vec4(rotatedPosition, 1.0);
        worldPosition.xyz += instanceOffset;
        worldNormal = normalMatrix * rotatedNormal;
    }
    gl_Position = cameraUBO.proj * cameraUBO.view * worldPosition;
    fragNormal = worldNormal;
    fragTexCoord = inTexCoord;
}
