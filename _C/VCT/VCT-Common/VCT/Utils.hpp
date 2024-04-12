#pragma once
#include <glm/glm.hpp>

namespace VCT::Utils
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

    inline __device__ glm::uvec3 GetVoxelCoord(const glm::vec3& position, const glm::vec3& voxelWorldOrigin, float voxelSize)
    {
        return glm::uvec3((position - voxelWorldOrigin) / voxelSize);
    }

    inline __device__ glm::uvec3 GetVoxelCoord(const glm::vec3& position, const VoxelWorldInfo& vwInfo)
    {
        return glm::uvec3((position - vwInfo.worldOrigin) / vwInfo.size);
    }

    inline __device__ uint32_t VoxelCoordToID(const uint3& voxelCoord, const uint3& dimensions)
    {
        return (voxelCoord.z * dimensions.x * dimensions.y) + (voxelCoord.y * dimensions.x) + voxelCoord.x;
    }

    inline __device__ uint32_t VoxelCoordToID(const glm::uvec3& voxelCoord, const glm::uvec3& dimensions)
    {
        return (voxelCoord.z * dimensions.x * dimensions.y) + (voxelCoord.y * dimensions.x) + voxelCoord.x;
    }

    inline __device__ uint32_t VoxelCoordToID(const int3& voxelCoord, const uint3& dimensions)
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

    template <typename Type>
    inline __device__ float DistanceSquared(const Type& a, const Type& b)
    {
        Type c = b - a;
        return dot(c, c);
    }

    inline __device__ bool IntersectAabb(const OptixAabb& a, const OptixAabb& b)
    {
        return (a.minX <= b.maxX && a.maxX >= b.minX) &&
            (a.minY <= b.maxY && a.maxY >= b.minY) &&
            (a.minZ <= b.maxZ && a.maxZ >= b.minZ);
    }

    inline __device__ OptixAabb CreateAabbForVoxel(glm::uvec3 voxelCoord, const VoxelWorldInfo& vwInfo, float bias = 0.0f)
    {
        OptixAabb result{};
        glm::vec3 wPos = VoxelToWorld(voxelCoord, vwInfo.worldOrigin, vwInfo.size, vwInfo.halfSize);

        result.maxX = wPos.x + vwInfo.halfSize + bias;
        result.maxY = wPos.y + vwInfo.halfSize + bias;
        result.maxZ = wPos.z + vwInfo.halfSize + bias;

        result.minX = wPos.x - vwInfo.halfSize - bias;
        result.minY = wPos.y - vwInfo.halfSize - bias;
        result.minZ = wPos.z - vwInfo.halfSize - bias;

        return result;
    }

    template<typename Vec3DType>
    inline __device__ bool IsFinite(const Vec3DType& v)
    {
        return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
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

    inline __device__ bool IsPointOnLine(const glm::vec3& lineStart, const glm::vec3& lineEnd, const glm::vec3& point)
    {
        glm::vec3 lineDir = glm::normalize(lineEnd - lineStart);
        glm::vec3 pointDir = glm::normalize(point - lineStart);
        float distToPoint = VCT::Utils::DistanceSquared(lineStart, point);
        float distToEnd = VCT::Utils::DistanceSquared(lineStart, lineEnd);
        return glm::dot(lineDir, pointDir) > 0.999f && distToEnd > distToPoint;
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
}