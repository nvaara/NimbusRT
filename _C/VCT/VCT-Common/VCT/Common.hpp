#pragma once
#include "Profiler.hpp"
#include <map>
#include "Types.hpp"

struct AABB
{
    glm::vec3 min;
    glm::vec3 max;
};

struct VCTParams
{
    float voxelSize = 0.5f;
    uint32_t blockSize = 32;
    std::vector<VCT::Transmitter> transmitters;
    std::vector<VCT::Receiver> receivers;
    uint32_t maximumNumberOfInteractions = 2;
    uint32_t maximumNumberOfDiffractions = 1;
    bool refractionsEnabled = false;
    uint32_t receivedPathBufferSize = 1 << 13;
    uint32_t propagationPathBufferSize = 1 << 17;
    float propagationBufferSizeIncreaseFactor = 2.0f;
    uint32_t ieVoxelAxisSizeFactor = 2;
    uint32_t subIeVoxelAxisSizeFactor = 4;
    bool useLabelHashing = true;
    bool useConeReflections = true;
    uint32_t numOfCoarsePathsPerUniqueRoute = 100;

    VCT::RefineParams refineParams = { 2000, 1e-4f, 0.4f, 0.4f, 0.99999f, 0.002f };
    float sampleRadiusCoarse = 0.015f;
    float sampleRadiusRefine = 0.005f;
    float varianceFactorCoarse = 2.0f;
    float varianceFactorRefine = 2.0f;
    float sdfThresholdCoarse = 0.0015f;
    float sdfThresholdRefine = 0.0005f;
    std::string outputFileStem = "vct-output";
};

inline std::ostream& operator<<(std::ostream& out, const glm::vec3& v)
{
    out << "[" << v.x << ", " << v.y << ", " << v.z << "]";
    return out;
}

template <typename Type, typename... Types>
inline void CombineHash(std::size_t& seed, const Type& hashable, const Types&... hashables)
{
    seed ^= std::hash<Type>()(hashable) + 0x9E3779B9 + (seed << 6) + (seed >> 2);
    if constexpr (sizeof...(Types) > 0)
        return CombineHash(seed, hashables...);
}

template <typename... Types>
size_t CalculateHash(const Types&... types)
{
    size_t hash = 0;
    CombineHash(hash, types...);
    return hash;
}

#define MAKE_HASHABLE(Type, ...) \
namespace std \
{ \
    template <> struct hash<Type> \
    { \
        size_t operator()(const Type& t) const\
    { \
        size_t hash = 0; \
        CombineHash(hash, __VA_ARGS__); \
        return hash; \
    } \
    }; \
}