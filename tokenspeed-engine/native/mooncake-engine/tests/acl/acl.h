#pragma once
#include <cstdint>
inline constexpr int ACL_SUCCESS = 0;
inline int device_binds = 0;
inline int aclrtSetDevice(int32_t) {
    ++device_binds;
    return ACL_SUCCESS;
}
inline int aclrtGetDevice(int32_t* device) {
    *device = 0;
    return ACL_SUCCESS;
}

enum aclrtMemLocationType {
    ACL_MEM_LOCATION_TYPE_HOST, ACL_MEM_LOCATION_TYPE_DEVICE,
    ACL_MEM_LOCATION_TYPE_UNREGISTERED, ACL_MEM_LOCATION_TYPE_HOST_NUMA = 4
};
struct aclrtPtrAttributes {
    struct { int id; aclrtMemLocationType type; } location;
};
inline aclrtMemLocationType fixture_location = ACL_MEM_LOCATION_TYPE_DEVICE;
inline int aclrtPointerGetAttributes(const void*, aclrtPtrAttributes* attrs) {
    attrs->location = {0, fixture_location};
    return ACL_SUCCESS;
}
