#pragma once
#include <glm/glm.hpp>
#include "Types.hpp"

namespace Nimbus::Utils
{
    inline uint32_t GetLaunchCount(uint32_t totalCount, uint32_t blockSize)
    {
        return totalCount / blockSize + (totalCount % blockSize != 0);
    }

    inline glm::uvec3 GetLaunchCount(const glm::uvec3& totalCounts, uint32_t blockSize)
    {
        return glm::uvec3(GetLaunchCount(totalCounts.x, blockSize),
                          GetLaunchCount(totalCounts.y, blockSize),
                          GetLaunchCount(totalCounts.z, blockSize));
    }

    inline __device__ glm::vec3 WorldToVoxel(const glm::vec3& position, const glm::vec3& voxelWorldOrigin, float invVoxelSize)
    {
        return (position - voxelWorldOrigin) * invVoxelSize;
    }

    inline __device__ glm::vec3 WorldToVoxel(const glm::vec3& position, const VoxelWorldInfo& vwInfo)
    {
        return (position - vwInfo.worldOrigin) * vwInfo.inverseSize;
    }

    inline __device__ glm::vec3 VoxelToWorld(const glm::vec3& position, const glm::vec3& voxelWorldOrigin, float voxelSize)
    {
        return voxelWorldOrigin + position * voxelSize;
    }

    inline __device__ glm::vec3 VoxelToWorld(const glm::uvec3& position, const glm::vec3& voxelWorldOrigin, float voxelSize, float halfVoxelSize)
    {
        return voxelWorldOrigin + glm::vec3(position) * voxelSize + halfVoxelSize;
    }

    inline __device__ glm::vec3 VoxelToWorld(const glm::uvec3& position, const VoxelWorldInfo& vwInfo)
    {
        return vwInfo.worldOrigin + glm::vec3(position) * vwInfo.size + vwInfo.halfSize;
    }

    inline __device__ glm::uvec3 GetVoxelCoord(const glm::vec3& position, const glm::vec3& voxelWorldOrigin, float voxelSize)
    {
        return glm::uvec3((position - voxelWorldOrigin) / voxelSize);
    }

    inline __device__ glm::uvec3 GetVoxelCoord(const glm::vec3& position, const VoxelWorldInfo& vwInfo)
    {
        return glm::uvec3((position - vwInfo.worldOrigin) / vwInfo.size);
    }

    inline __device__ uint32_t VoxelCoordToID(const glm::uvec3& voxelCoord, const glm::uvec3& dimensions)
    {
        return (voxelCoord.z * dimensions.x * dimensions.y) + (voxelCoord.y * dimensions.x) + voxelCoord.x;
    }

    inline __device__ uint32_t WorldToVoxelID(const glm::vec3& position, const glm::vec3& voxelWorldOrigin, float voxelSize, const glm::uvec3& dimensions)
    {
        glm::vec3 voxelCoord = GetVoxelCoord(position, voxelWorldOrigin, voxelSize);
        return VoxelCoordToID(voxelCoord, dimensions);
    }

    inline __device__ uint32_t WorldToVoxelID(const glm::vec3& position, const VoxelWorldInfo& vwInfo)
    {
        glm::vec3 voxelCoord = GetVoxelCoord(position, vwInfo);
        return VoxelCoordToID(voxelCoord, vwInfo.dimensions);
    }

    inline __device__ glm::uvec3 VoxelIDToCoord(uint32_t id, const glm::uvec3& dimensions)
    {
        glm::uvec3 result;
        uint32_t xy = dimensions.x * dimensions.y;
        result.z = id / xy;
        uint32_t idx2d = id - xy * result.z;
        result.x = idx2d % dimensions.x;
        result.y = idx2d / dimensions.x;
        return result;
    }

    inline __device__ glm::vec3 VoxelIDToWorld(uint32_t id, const glm::vec3& voxelWorldOrigin, float voxelSize, const glm::uvec3& dimensions)
    {
        glm::uvec3 coord = VoxelIDToCoord(id, dimensions);
        return VoxelToWorld(coord, voxelWorldOrigin, voxelSize, voxelSize / 2);
    }

    inline __device__ glm::vec3 VoxelIDToWorld(uint32_t id, const VoxelWorldInfo& vwInfo)
    {
        glm::uvec3 coord = VoxelIDToCoord(id, vwInfo.dimensions);
        return VoxelToWorld(coord, vwInfo.worldOrigin, vwInfo.size, vwInfo.halfSize);
    }

    inline __device__ bool IsIncident(const glm::vec3& v1, const glm::vec3& v2, float threshold = 0.99f)
    {
        return glm::abs(glm::dot(v1, v2)) > threshold;
    }

    inline __device__ void GetOrientationVectors(const glm::vec3& forward, glm::vec3& right, glm::vec3& up)
    {
        glm::vec3 globalUp = glm::vec3(0.0f, 0.0f, 1.0f);
        globalUp = IsIncident(forward, globalUp) ? glm::vec3(1.0f, 0.0f, 0.0f) : globalUp;
        right = glm::normalize(glm::cross(globalUp, forward));
        up = glm::cross(forward, right);
    }

    inline __device__ glm::vec3 FixNormal(const glm::vec3& incident, const glm::vec3& normal)
    {
        return glm::faceforward(normal, incident, normal);
    }

    inline __device__ bool IsPointInAabb(const OptixAabb& aabb, const glm::vec3& p, float bias = 0.0f)
    {
        return p.x >= aabb.minX - bias &&
               p.y >= aabb.minY - bias &&
               p.z >= aabb.minZ - bias &&
               p.x <= aabb.maxX + bias &&
               p.y <= aabb.maxY + bias &&
               p.z <= aabb.maxZ + bias;
    }

    inline __device__ glm::vec3 ProjectPointToRay(const glm::vec3& origin, const glm::vec3& direction, const glm::vec3& point)
    {
        return origin + glm::dot(point - origin, direction) * direction;
    }

    inline __device__ glm::uvec2 GetBinBitMask32(uint32_t index)
    {
        constexpr uint32_t bitsIn32 = 5u;
        uint32_t bin = index >> bitsIn32;
        uint32_t modulo32 = index & 31u;
        uint32_t bitMask = 1u << modulo32;
        return glm::uvec2(bin, bitMask);
    }

    template <typename PtrType>
    inline __device__ PtrType* UnpackPointer32(uint32_t i0, uint32_t i1)
    {
        return reinterpret_cast<PtrType*>(static_cast<uint64_t>(i0) << 32 | i1);
    }

    inline __device__ glm::uvec2 PackPointer32(const void* ptr)
    {
        uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        return glm::uvec2(uptr >> 32u, uptr & 0xFFFFFFFF);
    }
}