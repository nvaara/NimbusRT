#pragma once
#include <string_view>
#include "Types.hpp"
#include "Common.hpp"
#include "PathStorage.hpp"
#include <future>
#include "InputData.hpp"

namespace VCT
{
    class VoxelConeTracer
    {
    public:
        VoxelConeTracer();
        ~VoxelConeTracer();

        bool Prepare(const PointData* points, size_t numPoints, const std::vector<Edge>& edges, const VCTParams& params);
        bool Prepare(const PointData* points,
                     size_t numPoints,
                     const InputData& inputData,
                     const std::unordered_map<std::string, Object3D>& txs,
                     const std::unordered_map<std::string, Object3D>& rxs,
                     const std::vector<Edge>& edges);

        void Trace();
        void Refine(uint32_t txID, uint32_t rxID);
        const std::string& GetTransmitterName(uint32_t txID) const { return m_TxIDs.at(txID); }
        const std::string& GetReceiverName(uint32_t rxID) const { return m_RxIDs.at(rxID); }
        const PathStorage& GetRefinedPathStorage() const { return m_RefinedPathStorage; }

    private:
        const glm::vec3 GetWorldCenter() const { return (m_SceneAABB.max + m_SceneAABB.min) / 2.f; }
        const glm::uvec3& GetVoxelDimensions() const { return m_VoxelDimensions; }
        uint32_t GetVoxelCount() const { return m_VoxelDimensions.x * m_VoxelDimensions.y * m_VoxelDimensions.z; }

        float GetIeVoxelSize() const { return m_Params.voxelSize / m_Params.ieVoxelAxisSizeFactor; }
        uint32_t GetIeVoxelCount() const { auto pcdDim = GetIeVoxelDimensions(); return pcdDim.x * pcdDim.y * pcdDim.z; }
        uint32_t GetSubIeSizeFactor() const { return m_Params.subIeVoxelAxisSizeFactor * m_Params.subIeVoxelAxisSizeFactor * m_Params.subIeVoxelAxisSizeFactor; }
        uint32_t GetSubIeVoxelCount() const { return GetIeVoxelCount() * GetSubIeSizeFactor(); }
        glm::uvec3 GetIeVoxelDimensions() const { return m_VoxelDimensions * m_Params.ieVoxelAxisSizeFactor; }

        bool LoadPointCloud(const PointData* points, size_t numPoints, const std::vector<Edge>& edges);
        void LoadDiffractionEdges(const std::vector<Edge>& diffractionEdges);
        void LoadSurfacePoints(const PointData* points, size_t numPoints);
        void LoadEdgePoints();
        void LoadReceiverPoints();
        void CalculateVoxelDimensions();
        void LinkPointNodes();
        void UploadBuffers();
        void GenerateDataForRayTracing();
        void CreateVoxelTexture();
        void PrepareTrace(uint32_t transmitterID);
        void IncreaseDepth();
        void DecreaseDepth();
        void Transmit();
        void Propagate();
        void TraceTransmitter(uint32_t transmitterID);
        void CalculateDiffractionRays();
        VCTData CreateVCTData() const;
        void RetrievePaths(DeviceBuffer* deviceBuffer, uint32_t numPaths);
        void PostProcess(uint32_t txID, uint32_t rxID);

    private:
        bool m_Initialized;
        std::vector<PointNode> m_PointNodes;
        std::vector<uint2> m_IeVoxelNodeIndices;
        std::vector<uint2> m_VoxelTextureData;
        std::vector<uint32_t> m_PerIeSubIePrimitiveCount;
        std::vector<VoxelPointData> m_VoxelPointData;
        uint32_t m_NumberOfSurfacePoints;
        uint32_t m_IePrimitiveCount;
        uint32_t m_SubIePrimitiveCount;
        AABB m_SceneAABB;
        std::vector<std::string> m_TxIDs;
        std::vector<std::string> m_RxIDs;
        glm::uvec3 m_VoxelDimensions;
        VCTParams m_Params;
        Channel m_Channel;
        AccelerationStructure m_AccelerationStructure;

        DeviceBuffer m_PointNodeBuffer;
        DeviceBuffer m_IeVoxelPointNodeIndicesBuffer;
        DeviceBuffer m_VoxelTextureDataBuffer;
        DeviceBuffer m_VoxelPointDataBuffer;
        DeviceBuffer m_VoxelInfoBuffer;

        DeviceBuffer m_IePointBuffer;
        DeviceBuffer m_IePrimitiveBuffer;
        DeviceBuffer m_IntersectableEntityBuffer;
        DeviceBuffer m_IeCountBuffer;
        DeviceBuffer m_IePointCountBuffer;
        DeviceBuffer m_IePrimitiveCountBuffer;
        DeviceBuffer m_IePrimitiveInfoBuffer;

        DeviceBuffer m_SubIePrimitiveCountBuffer;
        DeviceBuffer m_SubIePrimitivePointCountBuffer;
        DeviceBuffer m_PerIeSubIePrimitiveCountBuffer;
        DeviceBuffer m_SubIePrimitivePointBuffer;
        DeviceBuffer m_SubIePrimitiveBuffer;
        DeviceBuffer m_SubIePrimitiveInfoBuffer;
        DeviceBuffer m_SubIePrimitiveVoxelMapBuffer;
        DeviceBuffer m_SubIePrimitiveNeighborsBuffer;

        DeviceBuffer m_NumReceivedPathsBuffer;
        std::vector<DeviceBuffer> m_PropPathBuffers;
        DeviceBuffer m_PropPathPointerBuffer;
        DeviceBuffer m_MaxNumPropPathBuffer;
        
        std::vector<DiffractionRay> m_DiffractionRays;
        std::vector<IndexInfo> m_DiffractionRayIndexInfos;
        DeviceBuffer m_DiffractionRayBuffer;
        DeviceBuffer m_DiffractionRayIndexInfoBuffer;
        DeviceBuffer m_DiffractionEdgeBuffer;
        DeviceBuffer m_DiffractionEdgeSegmentBuffer;

        float m_MaxDiffuseAngle;
        float m_DiffuseAngleSin;
        float m_DiffuseAngleCos;
        std::vector<DiffractionEdge> m_DiffractionEdges;
        std::vector<DiffractionEdgeSegment> m_DiffractionEdgeSegments;

        cudaTextureObject_t m_VoxelTexture;
        DeviceBuffer m_TransmitterBuffer;
        DeviceBuffer m_ReceiverBuffer;
        uint32_t m_IeCount;

        std::vector<uint32_t> m_MaxNumPropPaths;
        VCTData m_VCTData;
        DeviceBuffer m_VCTDataBuffer;

        DeviceBuffer m_PathProcessingDataBuffer;
        DeviceBuffer m_TransmitIndexProcessedBuffer;
        DeviceBuffer m_VCTStatusBuffer;
        DeviceBuffer m_DepthLevelBuffer;
        int32_t m_DepthLevel;
        Status m_TransmitStatus;
        struct PropagationStatus
        {
            bool ProcessingRequired() const { return status == Status::ProcessingRequired && launchCount > 0; }
            Status status = Status::ProcessingRequired;
            uint32_t launchCount = 0;
        };
        std::vector<PropagationStatus> m_PropagationStatuses;

        PathStorage m_CoarsePathStorage;
        PathStorage m_RefinedPathStorage;
        std::unique_ptr<TraceData[]> m_TransferHostBuffer;
        uint32_t m_ActiveRecvBufferIndex;
        std::future<void> m_TransferStatus;
        std::array<DeviceBuffer, 2> m_CoarsePathBuffers;
        DeviceBuffer m_ActiveBufferIndexBuffer;
        bool m_UseLabelHashing;
    };
}