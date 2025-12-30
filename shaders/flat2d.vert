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
    vec4 rotation; // xyz = Euler angles (radians), w = cap flag (unused in 2D path)
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
layout(location = 1) in vec3 inNormal; // Unused in 2D path, kept for binding compatibility
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;

void main() {
    uint instanceCount = max(objectUBO.instanceData.x, 1u);
    uint instanceIdx = min(gl_InstanceIndex, instanceCount - 1u);
    vec3 instanceOffset = vec3(0.0);
    vec3 instanceRotation = vec3(0.0);
    if (objectUBO.instanceData.x > 0u) {
        InstanceTransform inst = instanceBuffer.transforms[instanceIdx];
        instanceOffset = inst.offset.xyz;
        instanceRotation = inst.rotation.xyz;
    }

    vec3 rotatedPosition = rotateByEuler(inPosition, instanceRotation);
    vec4 worldPosition = objectUBO.model * vec4(rotatedPosition + instanceOffset, 1.0);

    gl_Position = cameraUBO.proj * cameraUBO.view * worldPosition;
    fragTexCoord = inTexCoord;
}
