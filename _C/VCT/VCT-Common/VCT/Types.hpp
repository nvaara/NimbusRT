#pragma once

#ifndef __CUDACC__
#include <iostream>
#include <array>
#define DEBUG_PRINT(...) std::cout << __VA_ARGS__ << std::endl;
#else
#define DEBUG_PRINT(...)
#endif

#include "Kernel.hpp"
#include "Propagation.hpp"
#include "Intersection.hpp"

namespace VCT
{
    struct Channel
    {
        __device__ Channel() = default;
        __device__ Channel(float frequency);

        float frequency;
        float waveLength;
        float waveNumber;
        float effectiveArea;
    };

    inline __device__ Channel::Channel(float frequency)
        : frequency(frequency)
        , waveLength(Constants::LightSpeedInVacuum / frequency)
        , waveNumber(2 * Constants::Pi / waveLength)
        , effectiveArea((waveLength* waveLength) / (Constants::Pi * 4))
    {
    }

    enum class IEType : uint32_t
    {
        Receiver = 0,
        Edge,
        Surface,
    };

    struct VoxelIENode
    {
        IEType type;
        uint32_t id;
        uint32_t next;
    };

    struct PointData
    {
        glm::vec3 position;
        glm::vec3 normal;
        uint32_t label;
        uint32_t material;
    };

    struct PointNode
    {
        glm::vec3 position;
        glm::vec3 normal;
        uint32_t label;
        IEType type;
        uint32_t ieNext;
        union
        {
            uint32_t receiverID;
            uint32_t edgeSegmentID;
            uint32_t materialID;
        };
    };

    struct PrimitivePoint
    { 
        glm::vec3 position;
        glm::vec3 normal;
        uint32_t label;
        uint32_t materialID;
    };

    struct DiffractionRay
    {
        glm::vec2 direction;
        glm::vec2 planeDirections[2];
    };

    struct IndexInfo
    {
        uint32_t first;
        uint32_t count;
    };

    template <typename T1, typename T2>
    inline __device__ float Cross2D(const T1& a, const T2& b)
    {
        return a.x * b.y - b.x * a.y;
    }

    struct DiffractionEdge
    {
        template <typename T1, typename T2, typename T3>
        static __device__ bool IsValidDiffractionRay(const T1& normal0, const T2& normal1, const T3& ray);
        __device__ bool IsValidIncidentRayForDiffraction(const glm::vec3& incidentRay) const;
        __device__ bool IsValidDiffractionRay(const DiffractionRay& ray) const;
        __device__ uint32_t GetNumberOfValidDiffractionRays(const DiffractionRay* diffractionRays, const IndexInfo& diffIndexInfo) const;

        glm::vec3 forward;
        glm::vec3 up;
        glm::vec3 right;
        glm::vec3 startPoint;
        glm::vec3 endPoint;
        glm::vec3 normal0;
        glm::vec3 normal1;
        glm::vec3 tangent0;
        glm::vec3 tangent1;
        glm::vec3 combinedNormal;
        float n;
        uint32_t materialID;
        glm::mat3 inverseMatrix;
        glm::vec2 localSurfaceDir2D0;
        glm::vec2 localSurfaceDir2D1;
        uint32_t firstInfoIndex;
    };

    template <typename T1, typename T2, typename T3>
    inline __device__ bool DiffractionEdge::IsValidDiffractionRay(const T1& dir0, const T2& dir1, const T3& ray)
    {
        bool test0 = Cross2D(dir0, ray) * Cross2D(dir0, dir1) >= 0;
        bool test1 = Cross2D(dir1, ray) * Cross2D(dir1, dir0) >= 0;
        return !(test0 && test1);
    }

    inline __device__ bool DiffractionEdge::IsValidIncidentRayForDiffraction(const glm::vec3& incidentRay) const
    {
        glm::vec3 localIncidentRay = inverseMatrix * incidentRay;
        glm::vec2 localIncident2D = glm::normalize(-glm::vec2(localIncidentRay.x, localIncidentRay.z));
        return IsValidDiffractionRay(localSurfaceDir2D0, localSurfaceDir2D1, localIncident2D);
    }

    inline __device__ bool DiffractionEdge::IsValidDiffractionRay(const DiffractionRay& ray) const
    {
        return IsValidDiffractionRay(localSurfaceDir2D0, localSurfaceDir2D1, ray.direction);
    }

    inline __device__ uint32_t DiffractionEdge::GetNumberOfValidDiffractionRays(const DiffractionRay* diffractionRays, const IndexInfo& diffIndexInfo) const
    {
        uint32_t count = 0;
        for (uint32_t rayIndex = 0; rayIndex < diffIndexInfo.count; ++rayIndex)
            count += IsValidDiffractionRay(diffractionRays[diffIndexInfo.first + rayIndex]);
        return count;
    }

    struct DiffractionEdgeSegment
    {
        uint32_t parentID;
        glm::vec3 startPoint;
        glm::vec3 endPoint;
        glm::vec3 startPointVoxelSpace;
        glm::vec3 endPointVoxelSpace;
    };

    struct DiffractionRayResult
    {
        glm::vec3 direction;
        glm::vec3 planeDirections[4];
        glm::vec3 planeNormals[4];
    };

    struct VoxelWorldInfo
    {
        __device__ VoxelWorldInfo() = default;
        __device__ VoxelWorldInfo(const glm::vec3& worldOrigin, float voxelSize, const glm::uvec3& dimensions)
            : worldOrigin(worldOrigin), size(voxelSize), halfSize(voxelSize * 0.5f), inverseSize(1.0f / voxelSize), dimensions(dimensions), count(dimensions.x * dimensions.y * dimensions.z) {}

        glm::vec3 worldOrigin;
        float size;
        float halfSize;
        float inverseSize;
        glm::uvec3 dimensions;
        uint32_t count;
    };

    struct VoxelInfo
    {
        glm::vec3 voxelSpaceCenter;
        IndexInfo ieIndexInfo;
    };

    struct VoxelPointData
    {
        uint32_t numPrimitives;
        uint32_t numSurfacePoints;
        uint32_t numEdges;
        uint32_t numReceivers;
    };

    struct PropagationPath
    {
        uint32_t iaCount;
        uint32_t pathPrimitiveIDs[Constants::MaximumNumberOfInteractions];
        uint32_t transmitterID;
        uint32_t receiverID;
    };

    struct IntersectableEntity
    {
        IEType type;
        glm::vec3 rtPoint;
        glm::vec3 voxelSpaceRtPoint;
        union
        {
            uint32_t receiverID;
            uint32_t edgeSegmentID;
            uint32_t surfaceLabel;
        };
    };

    struct IEPrimitiveInfo
    {
        IndexInfo pointIndexInfo;
        uint32_t ID;
    };

    struct PrimitiveNeighbors
    {
        uint8_t count;
        uint32_t neighbors[3 * 3 * 3 - 1];
    };

	struct VoxelizationData
	{
        VoxelWorldInfo voxelWorldInfo;
		PointNode* pointNodes;
        uint2* ieVoxelPointNodeIndices;
        uint2* voxelTextureData;
        VoxelWorldInfo ieVoxelWorldInfo;

        VoxelInfo* voxelInfos;
        PrimitivePoint* iePrimitivePoints;
        OptixAabb* iePrimitives;
        IntersectableEntity* intersectableEntities;
        IEPrimitiveInfo* iePrimitiveInfos;
        uint32_t* ieCount;
        uint32_t* iePointCount;
        uint32_t* iePrimitiveCount;
        uint32_t ieVoxelFactor;
        const VoxelPointData* voxelPointData;

        uint32_t subIe;
        VoxelWorldInfo subIeVoxelWorldInfo;
        uint32_t* subIePrimitiveCount;
        uint32_t* subIePrimitivePointCount;
        const uint32_t* perIeSubIePrimitiveCount;
        PrimitivePoint* subIePrimitivePoints;
        OptixAabb* subIePrimitives;
        IEPrimitiveInfo* subIePrimitiveInfos;
        uint32_t* subIePrimitiveVoxelMap;
        PrimitiveNeighbors* subIePrimitiveNeighbors;
	};

    template <InteractionType Type>
    inline constexpr __device__ bool IsInteractionType(const InteractionType& iaType)
    {
        std::underlying_type<InteractionType>::type expected = static_cast<std::underlying_type<InteractionType>::type>(Type);
        std::underlying_type<InteractionType>::type value = static_cast<std::underlying_type<InteractionType>::type>(iaType);
        return (expected & value) == expected;
    }

    struct IntersectionData
    {
        __device__ bool Intersect(const glm::vec3& point, float radius, float planeIntersectionRadius) const;
        Cone rayCone;
        Plane separationPlanes[2];
    };

    inline __device__ bool IntersectionData::Intersect(const glm::vec3& point, float radius, float planeIntersectionRadius) const
    {
        bool result = true;
        result &= rayCone.Intersect(point, radius, 0.0f);
        result &= separationPlanes[0].SignedDistance(point) <= planeIntersectionRadius;
        result &= separationPlanes[1].SignedDistance(point) <= planeIntersectionRadius;
        return result;
    }

    struct VoxelTraceData
    {
        glm::vec3 rayDirection;
        glm::vec3 voxel;
        IntersectionData intersectionData;
        uint32_t localIeID;
        glm::u16vec3 localVoxel;
        glm::vec3 previousVoxel;
        uint32_t voxelHistory[3][3][3];
        bool finished;
    };

    struct TraceProcessingData
    {
        TraceData traceData;
        uint32_t numDiffractions;
        float incidentIor;
    };

    struct PropagationData
    {
        TraceProcessingData tpData;
        VoxelTraceData voxelTraceData;
    };

    struct PathProcessingData
    {
        uint32_t numPaths;
        uint32_t numPathsToProcess;
        uint32_t nextNumPathsToProcess;
    };
    
    enum class Status : uint32_t
    {
        Finished = 0,
        ProcessingRequired
    };

    struct RefineParams
    {
        uint32_t numIterations;
        float delta;
        float beta;
        float alpha;
        float angleThreshold;
        float distanceThreshold;
    };

    struct RayTracingParams
    {
        OptixTraversableHandle asHandle;
        const OptixAabb* primitives;
        const PrimitivePoint* primitivePoints;
        const IEPrimitiveInfo* primitiveInfos;

        float sampleDistance;
        float sampleRadius;
        float traceDistanceBias;
        float varianceFactor;
        float sdfThreshold;
    };

    struct SceneData
    {
        const IntersectableEntity* intersectableEntities;
        const Transmitter* transmitters;
        const Receiver* receivers;
        RayTracingParams rtParams;
        RayTracingParams refineRtParams;
    };

    struct ConeTracingData
    {
        cudaTextureObject_t voxelTexture;
        const VoxelInfo* voxelInfos;
        VoxelWorldInfo voxelWorldInfo;
        float ieBoundingSphereRadius;
        const DiffractionEdge* diffractionEdges;
        const DiffractionEdgeSegment* diffractionEdgeSegments;
        const DiffractionRay* diffractionRays;
        const IndexInfo* diffractionIndexInfos;
        float sinDiffuseAngle;
        float cosDiffuseAngle;
        float reflSinDiffuseAngle;
        float reflCosDiffuseAngle;
        uint32_t maximumNumberOfInteractions;
        uint32_t maximumNumberOfDiffractions;
        const int32_t* depthLevel;
    };

    struct PathData
    {
        uint32_t maxNumReceivedPaths;
        uint32_t* numReceivedPaths;
        TraceData* coarsePaths[2];
        uint32_t* activeBufferIndex;
        PropagationData** propPaths;
        uint32_t* maxNumPropPaths;
        PathProcessingData* pathProcessingData;
    };

    struct VCTData
    {
        uint32_t currentTransmitterID;
        SceneData sceneData;
        ConeTracingData coneTracingData;
        PathData pathData;
        uint8_t* transmitIndexProcessed;
        Status* status;
        TraceData* pathsToRefine;
        TraceData* refinedPaths;
        uint32_t* numRefinedPaths;
        PrimitiveNeighbors* subIePrimitiveNeighbors;
        RefineParams refineParams;
    };
}