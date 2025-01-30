#pragma once
#include "Constants.hpp"
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <optix.h>
#include <vector>
#include <glm/glm.hpp>

namespace Nimbus
{
    struct Aabb
    {
        glm::vec3 min;
        glm::vec3 max;
    };

    enum class PathType : uint8_t
    {
        LineOfSight = 0x1,
        Specular = 0x2,
        Scattering = 0x4,
        Diffraction = 0x8,
        RIS = 0x10,
    };

    struct PathInfo
    {
        uint8_t numInteractions;
        PathType pathType;
        uint32_t txID;
        uint32_t rxID;
        double timeDelay;
    };

    struct PathInfoST : public PathInfo
    {
        uint32_t rtPointIndex;
        glm::vec3 direction;
        uint8_t terminated;
    };

    struct PointData
    {
        glm::vec3 position;
        glm::vec3 normal;
        uint32_t label;
        uint32_t material;
    };

    struct EdgeData
    {
        glm::vec3 start;
        glm::vec3 end;
        glm::vec3 normal1;
        glm::vec3 normal2;
        uint32_t material1;
        uint32_t material2;
        uint32_t edgeID;
    };

    struct Face
    {
        glm::vec3 normal;
        uint32_t label;
        uint32_t material;
    };

    struct PointNode
    {
        glm::vec3 position;
        glm::vec3 normal;
        uint32_t label;
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

    struct IndexInfo
    {
        uint32_t first;
        uint32_t count;
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

    struct IEPrimitiveInfo
    {
        IndexInfo pointIndexInfo;
        uint32_t ID;
    };

    struct RefineParams
    {
        uint32_t maxNumIterations;
        uint32_t maxCorrectionIterations;
        float delta;
        float beta;
        float alpha;
        float angleThreshold;
        float distanceThreshold;
    };

    struct DiffractionEdge
    {
        __device__ static float Cross2D(const glm::vec2& a, const glm::vec2& b);
        __device__ bool IsValidIncidentRayForDiffraction(const glm::vec3& incidentRay) const;

        glm::vec3 forward;
        glm::vec3 start;
        glm::vec3 end;
        glm::vec3 midPoint;
        float halfLength;
        glm::vec3 normal0;
        glm::vec3 normal1;
        glm::mat3 inverseMatrix;
        glm::vec2 localSurfaceDir2D0;
        glm::vec2 localSurfaceDir2D1;
    };

    inline __device__ float DiffractionEdge::Cross2D(const glm::vec2& a, const glm::vec2& b)
    {
        return a.x * b.y - b.x * a.y;
    }

    inline __device__ bool DiffractionEdge::IsValidIncidentRayForDiffraction(const glm::vec3& incidentRay) const
    {
        glm::vec3 localIncidentRay = inverseMatrix * incidentRay;
        glm::vec2 localIncident2D = glm::normalize(-glm::vec2(localIncidentRay.x, localIncidentRay.z));

        bool test0 = Cross2D(localSurfaceDir2D0, localIncident2D) * Cross2D(localSurfaceDir2D0, localSurfaceDir2D1) >= 0;
        bool test1 = Cross2D(localSurfaceDir2D1, localIncident2D) * Cross2D(localSurfaceDir2D1, localSurfaceDir2D0) >= 0;
        return !(test0 && test1);
    }

    struct EnvironmentData
    {
        OptixTraversableHandle asHandle;
        const glm::vec3* rtPoints;
        VoxelWorldInfo vwInfo;
        const DiffractionEdge* edges;
        uint32_t edgeCount;
        union
        {
            struct
            {
                const OptixAabb* primitives;
                const PrimitivePoint* primitivePoints;
                const IEPrimitiveInfo* primitiveInfos;
            } pc;
            struct
            {
                bool useFaceNormals;
                const uint32_t* indices;
                const glm::vec3* normals;
                const Face* faces;
                const uint32_t* voxelToRtPointIndexMap;
            } triangle;
        };
    };

    struct RayTracingParams
    {
        EnvironmentData env;
        float sampleDistance;
        float sampleRadius;
        float varianceFactor;
        float sdfThreshold;

        float rayMaxLength;
        float rayBias;
    };

    struct STData
    {
        VoxelWorldInfo voxelWorldInfo;
        OptixAabb* primitives;
        glm::vec3* rtPoints;
        uint32_t* primitiveCount;
        PointNode* pointNodes;
        glm::uvec2* voxelPointNodeIndices;
        IEPrimitiveInfo* primitiveInfos;
        PrimitivePoint* points;
        uint32_t* pointCount;
        uint32_t ieCount;
        float aabbBias;
    };

    struct STRTData
    {
        uint32_t currentTxID;
        const glm::vec3* transmitters;
        const glm::vec3* receivers;
        RayTracingParams rtParams;
        RefineParams refineParams;
        uint32_t* scattererVisible;
        uint32_t* rxVisible;
        uint32_t numRx;

        uint32_t* pathsProcessed;
        uint32_t* pathCount;
        
        struct
        {
            PathInfo* pathInfos;
            glm::vec3* interactions;
            glm::vec3* normals;
            uint32_t* labels;
            uint32_t* materials;
        } receivedPathData;

        struct
        {
            PathInfoST* pathInfos;
            glm::vec3* interactions;
            glm::vec3* normals;
            uint32_t* labels;
            uint32_t* materials;
        } propagationPathData;

        struct
        {
            PathInfoST* pathInfos;
            glm::vec3* interactions;
            glm::vec3* normals;
            uint32_t* labels;
            uint32_t* materials;
        } diffractionPathData;

        uint32_t maxNumIa;
        uint32_t* receivedPathCount;
        uint32_t numReceivedPathsMax;
    };

    struct TriangleData
    {
        uint32_t numFaces;
        const glm::uvec3* indices;
        const glm::vec3* vertices;
        const Face* faces;
        VoxelWorldInfo vwInfo;

        glm::vec3* rtPoints;
        uint32_t* rtPointCounter;
        uint32_t* voxelToRtPointIndexMap;
    };

    struct CoverageData
    {
        uint32_t* cellProcessed;
        Nimbus::VoxelWorldInfo coverageWorldInfo;
        glm::vec3* outReceivers;
        glm::uvec2* outRx2D;
        uint32_t* outNumRx;
        const glm::vec3* rtPoints;
        uint32_t numPoints;
        float height;
    };

    struct CoverageMapInfo
    {
        CoverageMapInfo() : dimensions(0u), size(0.0f), height(0.0f) {}
        CoverageMapInfo(const glm::uvec2& dimensions, std::vector<glm::uvec2>&& rxCoords2D, float size, float height)
            : dimensions(dimensions), rxCoords2D(std::move(rxCoords2D)), size(size), height(height) {}
        
        glm::uvec2 dimensions;
        std::vector<glm::uvec2> rxCoords2D;
        float size;
        float height;
    };

    struct ScatterTracingParams
    {
        uint32_t maxNumInteractions;
        bool scattering;
        bool diffraction;
        float sampleRadius;
        float varianceFactor;
        float sdfThreshold;
        uint32_t numRefineIterations;
        uint32_t refineMaxCorrectionIterations;
        float refineDelta;
        float refineBeta;
        float refineAlpha;
        float refineAngleDegreesThreshold;
        float refineDistanceThreshold;
        float rayBias;
    };
}