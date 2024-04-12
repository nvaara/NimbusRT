#include "VoxelConeTracer.hpp"
#include "Utils.hpp"
#include "Traversal.hpp"
#include <numeric>
#include <future>
#include <filesystem>
#include <fstream>
#include "KernelData.hpp"

namespace
{
    #define ASSERT_VCT_PARAM(Result, ExpectedCondition, ...) if (!(ExpectedCondition)) { LOG(__VA_ARGS__); Result = false; }

    bool ValidateParams(const VCTParams& params)
    {
        bool result = true;
        ASSERT_VCT_PARAM(result, params.voxelSize > 0, "VCTParams: Voxel size should be > 0. Current: %f", params.voxelSize);
        ASSERT_VCT_PARAM(result, params.transmitters.size(), "VCTParams: Missing transmitter(s).");
        ASSERT_VCT_PARAM(result, params.receivers.size(), "VCTParams: Missing receiver(s).");
        ASSERT_VCT_PARAM(result, params.maximumNumberOfInteractions <= VCT::Constants::MaximumNumberOfInteractions, "Maximum number of interactions is too high. Interaction limit is 8.");
        ASSERT_VCT_PARAM(result, params.maximumNumberOfDiffractions <= params.maximumNumberOfInteractions, "Too many diffractions compared to maximum interaction limit.");
        ASSERT_VCT_PARAM(result, params.receivedPathBufferSize > 0, "ReceivedPathBufferSize should be greater than 0");
        ASSERT_VCT_PARAM(result, params.propagationPathBufferSize > 0, "PropagationPathBufferSize should be greater than 0");
        ASSERT_VCT_PARAM(result, params.propagationBufferSizeIncreaseFactor >= 1.0f, "PropagationBufferSizeIncreaseFactor should be >= 1");
        ASSERT_VCT_PARAM(result, params.refineParams.distanceThreshold >= 0.0f, "DistanceThreshold should be >= 0");
        ASSERT_VCT_PARAM(result, params.refineParams.angleThreshold >= 0.0f, "AngleThreshold should be >= 0");
        ASSERT_VCT_PARAM(result, params.refineParams.alpha > 0.0f && params.refineParams.alpha < 1.0, "Alpha should be greater than 0 and less than 1");
        ASSERT_VCT_PARAM(result, params.refineParams.beta > 0.0f && params.refineParams.beta < 1.0, "Beta should be greater than 0 and less than 1");
        ASSERT_VCT_PARAM(result, params.refineParams.numIterations > 0, "NumIterations should be greater than 0");
        ASSERT_VCT_PARAM(result, params.refineParams.delta >= 0.0f, "Delta should be >= 0");
        return result;
    }

    inline bool IsValidNormal(const glm::vec3& normal)
    {
        float distSq = dot(normal, normal);
        return !(isnan(distSq) || isinf(distSq) || distSq < 0.99f || distSq > 1.01f);
    }
    
    float FindAngleRads(const glm::vec2& vec)
    {
        float radsCos = std::acos(vec.y);
        float radsSin = std::asin(vec.x);

        for (int i = 0; i < 4; ++i)
        {
            float currentRadsCos = radsCos + glm::pi<float>() / 2 * i;
            float currentRadsSin = radsSin + glm::pi<float>() / 2 * i;
            glm::vec2 vCos = glm::vec2(std::sinf(currentRadsCos), std::cosf(currentRadsCos));
            glm::vec2 vSin = glm::vec2(std::sinf(currentRadsSin), std::cosf(currentRadsSin));

            if (dot(vec, vSin) > 0.99f)
                return currentRadsSin;

            if (dot(vec, vCos) > 0.99f)
                return currentRadsCos;
        }
        return 0.0f;
    }

    float FindDiffractionAngleRadsFromNormals(const glm::vec2& n0, const glm::vec2& n1)
    {
        float rads1 = FindAngleRads(n0);
        float rads0 = FindAngleRads(n1);

        if (rads1 > rads0)
            return (rads1 - rads0 < glm::pi<float>()) ? rads1 : rads0;

        return (rads0 - rads1 < glm::pi<float>() ? rads0 : rads1);
    }
}

namespace VCT
{
    VoxelConeTracer::VoxelConeTracer()
        : m_Initialized(false)
        , m_NumberOfSurfacePoints(0)
        , m_IePrimitiveCount(0)
        , m_SubIePrimitiveCount(0)
        , m_SceneAABB({})
        , m_Params({})
        , m_VoxelDimensions(glm::vec3(0))
        , m_VCTData({})
        , m_ActiveRecvBufferIndex(0)
        , m_UseLabelHashing(false)
        , m_RefinedPathStorage(1)
        , m_Channel({})
        , m_DepthLevel(0)
        , m_DiffuseAngleCos(0)
        , m_DiffuseAngleSin(0)
        , m_MaxDiffuseAngle(0)
        , m_IeCount(0)
        , m_TransmitStatus(VCT::Status::ProcessingRequired)
        , m_VoxelTexture(0)
    {

    }

    VoxelConeTracer::~VoxelConeTracer()
    {

    }

    bool VoxelConeTracer::Prepare(const PointData* points, size_t numPoints, const std::vector<Edge>& edges, const VCTParams& params)
    {
        PROFILE_SCOPE();
        {
            if (m_Initialized)
            {
                LOG("VoxelConeTracer::Prepare: Already initialized.");
                return m_Initialized;
            }
            m_Params = params;
            if (!ValidateParams(m_Params) || !LoadPointCloud(points, numPoints, edges))
                return m_Initialized;
            
            CalculateDiffractionRays();
            CalculateVoxelDimensions();
            LinkPointNodes();
            UploadBuffers();
            m_Initialized = true;
        }
        {
            GenerateDataForRayTracing();
            m_AccelerationStructure = AccelerationStructure::CreateFromAabbs(m_SubIePrimitiveBuffer, m_SubIePrimitiveCount);
            if (!m_AccelerationStructure)
            {
                LOG("Failed to build acceleration structure.\n");
                m_Initialized = false;
            }
        }
        if (m_Initialized)
        {
            m_VCTData = CreateVCTData();
            m_VCTDataBuffer = DeviceBuffer(sizeof(VCTData));
            m_VCTDataBuffer.Upload(&m_VCTData, 1);
            m_TransferHostBuffer = std::make_unique<TraceData[]>(m_Params.receivedPathBufferSize);
            m_CoarsePathStorage = PathStorage(m_Params.numOfCoarsePathsPerUniqueRoute);
        }
        return m_Initialized;
    }

    bool VoxelConeTracer::Prepare(const PointData* points,
                                  size_t numPoints,
                                  const InputData& inputData,
                                  const std::unordered_map<std::string, Object3D>& txs,
                                  const std::unordered_map<std::string, Object3D>& rxs,
                                  const std::vector<Edge>& edges)
    {
        VCTParams params;
        params.voxelSize = inputData.sceneSettings.voxelSize;
        params.blockSize = inputData.sceneSettings.blockSize;

        for (auto& tx : txs)
        {
            params.transmitters.emplace_back(tx.second.position);
            m_TxIDs.push_back(tx.first);
        }
        for (auto& rx : rxs)
        {
            params.receivers.emplace_back(rx.second.position);
            m_RxIDs.push_back(rx.first);
        }
        params.maximumNumberOfInteractions = inputData.numInteractions;
        params.maximumNumberOfDiffractions = inputData.numDiffractions;
        params.refractionsEnabled = false;
        params.receivedPathBufferSize = inputData.sceneSettings.receivedPathBufferSize;
        params.propagationPathBufferSize = inputData.sceneSettings.propagationPathBufferSize;
        params.propagationBufferSizeIncreaseFactor = inputData.sceneSettings.propagationBufferSizeIncreaseFactor;
        params.ieVoxelAxisSizeFactor = inputData.sceneSettings.voxelDivisionFactor;
        params.subIeVoxelAxisSizeFactor = inputData.sceneSettings.subvoxelDivisionFactor;
        params.useLabelHashing = true;
        params.useConeReflections = true;
        params.numOfCoarsePathsPerUniqueRoute = inputData.sceneSettings.numCoarsePathsPerUniqueRoute;

        params.refineParams.numIterations = inputData.sceneSettings.numIterations;
        params.refineParams.delta = inputData.sceneSettings.delta;
        params.refineParams.beta = inputData.sceneSettings.beta;
        params.refineParams.alpha = inputData.sceneSettings.alpha;
        params.refineParams.angleThreshold = glm::radians(inputData.sceneSettings.angleThreshold);
        params.refineParams.distanceThreshold = inputData.sceneSettings.distanceThreshold;

        params.sampleRadiusCoarse = inputData.sceneSettings.sampleRadiusCoarse;
        params.sampleRadiusRefine = inputData.sceneSettings.sampleRadiusRefine;
        params.varianceFactorCoarse = inputData.sceneSettings.varianceFactorCoarse;
        params.varianceFactorRefine = inputData.sceneSettings.varianceFactorRefine;
        params.sdfThresholdCoarse = inputData.sceneSettings.sdfThresholdCoarse;
        params.sdfThresholdRefine = inputData.sceneSettings.sdfThresholdRefine;
        params.outputFileStem = "vct-output";

        return Prepare(points, numPoints, edges, params);
    }

    void VoxelConeTracer::Trace()
    {
        PROFILE_SCOPE();
        if (!m_Initialized)
        {
            LOG("%s: Not initialized. Forgot to call VoxelConeTracer::Prepare?", __func__);
            return;
        }

        for (uint32_t transmitterID = 0; transmitterID < static_cast<uint32_t>(m_Params.transmitters.size()); ++transmitterID)
            TraceTransmitter(transmitterID);
    }

    void VoxelConeTracer::Refine(uint32_t txID, uint32_t rxID)
    {
        {
            PROFILE_SCOPE();
            const std::vector<TraceData>* paths = m_CoarsePathStorage.GetPaths(txID, rxID);
            if (!paths)
            {
                LOG("No paths to refine.");
                return;
            }
            DeviceBuffer pathsToRefineBuffer = DeviceBuffer::Create(*paths);
            DeviceBuffer refinedPathsBuffer = DeviceBuffer(sizeof(TraceData) * paths->size());
            DeviceBuffer numRefinedPathsBuffer = DeviceBuffer(sizeof(uint32_t));
            numRefinedPathsBuffer.MemsetZero();

            m_VCTData.pathsToRefine = pathsToRefineBuffer.DevicePointerCast<TraceData>();
            m_VCTData.refinedPaths = refinedPathsBuffer.DevicePointerCast<TraceData>();
            m_VCTData.numRefinedPaths = numRefinedPathsBuffer.DevicePointerCast<uint32_t>();

            m_VCTDataBuffer.Upload(&m_VCTData, 1);
            KernelData::Get().GetRefinePipeline().LaunchAndSynchronize(m_VCTDataBuffer, glm::uvec3(paths->size(), 1, 1));
            uint32_t numRefined = 0;
            numRefinedPathsBuffer.Download(&numRefined, 1);
            LOG("Number of refined paths that converged: %u", numRefined);
            if (numRefined)
            {
                std::vector<TraceData> refinedPaths(numRefined);
                refinedPathsBuffer.Download(refinedPaths.data(), refinedPaths.size());
                m_RefinedPathStorage.AddPaths(refinedPaths, m_UseLabelHashing);
            }
            else
                return;
        }
        PostProcess(txID, rxID);        
    }

    bool VoxelConeTracer::LoadPointCloud(const PointData* points, size_t numPoints, const std::vector<Edge>& edges)
    {
        m_UseLabelHashing = true;
        LoadDiffractionEdges(edges);
        m_PointNodes.reserve(numPoints + m_Params.receivers.size() + m_DiffractionEdgeSegments.capacity());
        m_SceneAABB.min = reinterpret_cast<const glm::vec3&>(points->position.x);
        m_SceneAABB.max = reinterpret_cast<const glm::vec3&>(points->position.x);
        LoadSurfacePoints(points, numPoints);
        LoadEdgePoints();
        LoadReceiverPoints();
        return true;
    }

    void VoxelConeTracer::LoadDiffractionEdges(const std::vector<Edge>& diffractionEdges)
    {
        for (const Edge& e : diffractionEdges)
        {
            DiffractionEdge edge{};
            edge.forward = glm::normalize(e.end - e.start);
            Utils::GetOrientationVectors(edge.forward, edge.right, edge.up);
            edge.startPoint = e.start;
            edge.endPoint = e.end;
            edge.normal0 = e.normal0;
            edge.normal1 = e.normal1;
            edge.materialID = 0; //Remove

            edge.inverseMatrix = glm::inverse(glm::mat3(edge.right, edge.forward, edge.up));

            glm::vec3 lerpNormal = glm::normalize(glm::mix(e.normal0, e.normal1, 0.5f));
            glm::vec3 n0 = glm::normalize(glm::cross(edge.forward, e.normal0));
            glm::vec3 n1 = glm::normalize(glm::cross(edge.forward, e.normal1));
            n0 = glm::dot(lerpNormal, n0) < 0.0f ? n0 : -n0;
            n1 = glm::dot(lerpNormal, n1) < 0.0f ? n1 : -n1;
            edge.combinedNormal = lerpNormal;

            edge.tangent0 = glm::normalize(glm::cross(edge.normal0, edge.forward));
            edge.tangent0 = glm::dot(edge.tangent0, lerpNormal) < 0.0f ? edge.tangent0 : -edge.tangent0;
            edge.tangent1 = glm::normalize(glm::cross(edge.normal1, edge.forward));
            edge.tangent1 = glm::dot(edge.tangent1, lerpNormal) < 0.0f ? edge.tangent1 : -edge.tangent1;

            edge.n = 2.0f - glm::acos(glm::dot(n0, n1)) / Constants::Pi;
            glm::vec3 localSurfaceDir2D0 = edge.inverseMatrix * n0;
            glm::vec3 localSurfaceDir2D1 = edge.inverseMatrix * n1;

            edge.localSurfaceDir2D0 = glm::normalize(glm::vec2(localSurfaceDir2D0.x, localSurfaceDir2D0.z));
            edge.localSurfaceDir2D1 = glm::normalize(glm::vec2(localSurfaceDir2D1.x, localSurfaceDir2D1.z));

            m_DiffractionEdges.push_back(edge);
            m_SceneAABB.min = glm::min(m_SceneAABB.min, e.start);
            m_SceneAABB.max = glm::max(m_SceneAABB.max, e.start);
            m_SceneAABB.min = glm::min(m_SceneAABB.min, e.end);
            m_SceneAABB.max = glm::max(m_SceneAABB.max, e.end);
        }
    }

    void VoxelConeTracer::LoadSurfacePoints(const PointData* points, size_t numPoints)
    {
        for (size_t i = 0; i < numPoints; ++i)
        {
            const PointData& point = points[i];
            if (IsValidNormal(point.normal))
            {
                ++m_NumberOfSurfacePoints;
                PointNode node{};
                node.position = point.position;
                node.normal = point.normal;
                node.label = point.label;
                node.materialID = point.material;
                node.type = IEType::Surface;
                node.ieNext = Constants::InvalidPointIndex;
                node.materialID = 1;
                m_PointNodes.push_back(node);

                m_SceneAABB.min = glm::min(m_SceneAABB.min, point.position);
                m_SceneAABB.max = glm::max(m_SceneAABB.max, point.position);
            }
        }
        LOG("Number of surface points: %u", m_NumberOfSurfacePoints);
    }

    void VoxelConeTracer::LoadEdgePoints()
    {
        float ieVoxelSize = GetIeVoxelSize();
        float halfieVoxelSize = ieVoxelSize / 2;
        glm::vec3 voxelWorldOrigin = m_SceneAABB.min;
        float invVoxelSize = 1.0f / m_Params.voxelSize;
        float invIeVoxelSize = 1.0f / ieVoxelSize;

        uint32_t parentID = 0;
        for (const DiffractionEdge& edge : m_DiffractionEdges)
        {
            float edgeLengthSq = Utils::DistanceSquared(edge.startPoint, edge.endPoint);
            VoxelTraverser traverser = VoxelTraverser(Utils::WorldToVoxel(edge.startPoint, voxelWorldOrigin, invIeVoxelSize), edge.forward);
            glm::vec3 previousPosition = edge.startPoint;
            
            bool edgeProcessingFinished = false;
            uint32_t nSegments = 0;
            while (!edgeProcessingFinished)
            {
                traverser.Step(0);
                glm::vec3 worldPosition = Utils::VoxelToWorld(traverser.GetTraverseVoxel(), voxelWorldOrigin, ieVoxelSize);
                float lengthSq = Utils::DistanceSquared(edge.startPoint, worldPosition);
                if (edgeLengthSq < lengthSq)
                {
                    worldPosition = edge.endPoint;
                    edgeProcessingFinished = true;
                }
                
                uint32_t segmentID = static_cast<uint32_t>(m_DiffractionEdgeSegments.size());
                DiffractionEdgeSegment segment{};
                segment.startPoint = previousPosition;
                segment.endPoint = worldPosition;
                segment.startPointVoxelSpace = Utils::WorldToVoxel(segment.startPoint, voxelWorldOrigin, invVoxelSize);
                segment.endPointVoxelSpace = Utils::WorldToVoxel(segment.endPoint, voxelWorldOrigin, invVoxelSize);
                segment.parentID = parentID;
                m_DiffractionEdgeSegments.push_back(segment);

                PointNode node{};
                node.position = (segment.startPoint + segment.endPoint) / 2.0f;
                node.type = IEType::Edge;
                node.ieNext = Constants::InvalidPointIndex;
                node.edgeSegmentID = segmentID;
                node.label = parentID;
                m_PointNodes.push_back(node);
                
                previousPosition = worldPosition;
            }
            parentID++;
        }
        LOG("Number of edge segments: %u", m_DiffractionEdgeSegments.size());
    }

    void VoxelConeTracer::LoadReceiverPoints()
    {
        uint32_t receiverID = 0;
        for (const Receiver& receiver : m_Params.receivers)
        {
            PointNode node{};
            node.position = receiver.position;
            node.type = IEType::Receiver;
            node.ieNext = Constants::InvalidPointIndex;
            node.receiverID = receiverID++;
            m_PointNodes.push_back(node);
            m_SceneAABB.min = glm::min(m_SceneAABB.min, node.position);
            m_SceneAABB.max = glm::max(m_SceneAABB.max, node.position);
        }
        LOG("Number of receivers %u", m_Params.receivers.size());
    }

    void VoxelConeTracer::CalculateVoxelDimensions()
    {
        m_VoxelDimensions.x = glm::max(static_cast<uint32_t>(std::ceil((m_SceneAABB.max.x - m_SceneAABB.min.x) / m_Params.voxelSize)), 1u);
        m_VoxelDimensions.y = glm::max(static_cast<uint32_t>(std::ceil((m_SceneAABB.max.y - m_SceneAABB.min.y) / m_Params.voxelSize)), 1u);
        m_VoxelDimensions.z = glm::max(static_cast<uint32_t>(std::ceil((m_SceneAABB.max.z - m_SceneAABB.min.z) / m_Params.voxelSize)), 1u);
    }

    void VoxelConeTracer::LinkPointNodes()
    {
        PROFILE_SCOPE();
        glm::vec3 voxelWorldOrigin = m_SceneAABB.min;
        glm::uvec3 ieVoxelDimensions = GetIeVoxelDimensions();
        float voxelSize = m_Params.voxelSize;
        float ieVoxelSize = GetIeVoxelSize();
        m_IeVoxelNodeIndices.resize(GetIeVoxelCount(), { Constants::InvalidPointIndex, 0 });
        m_PerIeSubIePrimitiveCount.resize(GetIeVoxelCount(), 0);
        std::vector<bool> refinePrimitiveCount(GetSubIeVoxelCount(), false);
        m_VoxelTextureData.resize(GetVoxelCount(), { Constants::InvalidPointIndex, Constants::InvalidPointIndex });
        m_VoxelPointData.resize(GetVoxelCount(), {});

        LOG("Number of point nodes : %u, SurfacePoints: %u", m_PointNodes.size(), m_NumberOfSurfacePoints);
        for (uint32_t pointIndex = 0; pointIndex < m_PointNodes.size(); ++pointIndex)
        {
            PointNode& pointNode = m_PointNodes[pointIndex];
            uint32_t voxelID = Utils::WorldToVoxelID(pointNode.position, voxelWorldOrigin, m_Params.voxelSize, m_VoxelDimensions);
            m_VoxelTextureData[voxelID].x = voxelID;
            VoxelPointData& vpData = m_VoxelPointData[voxelID];
            uint32_t ieVoxelID = Utils::WorldToVoxelID(pointNode.position, voxelWorldOrigin, ieVoxelSize, ieVoxelDimensions);
            uint32_t refineVoxelID = Utils::WorldToVoxelID(pointNode.position, voxelWorldOrigin, ieVoxelSize / m_Params.subIeVoxelAxisSizeFactor, ieVoxelDimensions * m_Params.subIeVoxelAxisSizeFactor);

            switch (pointNode.type)
            {
            case IEType::Surface:
            {
                ++vpData.numSurfacePoints;
                ++m_IeVoxelNodeIndices[ieVoxelID].y;
                if (m_IeVoxelNodeIndices[ieVoxelID].x == Constants::InvalidPointIndex)
                {
                    ++m_IePrimitiveCount;
                    ++vpData.numPrimitives;
                }
                if (!refinePrimitiveCount[refineVoxelID])
                {
                    refinePrimitiveCount[refineVoxelID] = true;
                    ++m_SubIePrimitiveCount;
                    ++m_PerIeSubIePrimitiveCount[ieVoxelID];
                }
                break;
            }
            case IEType::Receiver:
            {
                ++vpData.numReceivers;
                break;
            }
            case IEType::Edge:
            {
                ++vpData.numEdges;
                break;
            }
            default:
                break;
            }            
            pointNode.ieNext = m_IeVoxelNodeIndices[ieVoxelID].x;
            m_IeVoxelNodeIndices[ieVoxelID].x = pointIndex;
        }
    }

    void VoxelConeTracer::UploadBuffers()
    {
        m_PointNodeBuffer = DeviceBuffer::Create(m_PointNodes);

        m_IeVoxelPointNodeIndicesBuffer = DeviceBuffer::Create(m_IeVoxelNodeIndices);
        m_VoxelTextureDataBuffer = DeviceBuffer::Create(m_VoxelTextureData);

        VoxelizationData data{};
        glm::vec3 voxelWorldOrigin = m_SceneAABB.min;
        data.voxelWorldInfo = VoxelWorldInfo(m_SceneAABB.min, m_Params.voxelSize, m_VoxelDimensions);

        data.pointNodes = m_PointNodeBuffer.DevicePointerCast<PointNode>();
        data.ieVoxelPointNodeIndices = m_IeVoxelPointNodeIndicesBuffer.DevicePointerCast<uint2>();

        data.voxelTextureData = m_VoxelTextureDataBuffer.DevicePointerCast<uint2>();
        data.ieVoxelWorldInfo = VoxelWorldInfo(m_SceneAABB.min, GetIeVoxelSize(), GetIeVoxelDimensions());

        m_VoxelInfoBuffer = DeviceBuffer(sizeof(VoxelInfo) * GetVoxelCount());

        data.voxelInfos = m_VoxelInfoBuffer.DevicePointerCast<VoxelInfo>();

        m_IePointBuffer = DeviceBuffer(m_NumberOfSurfacePoints * sizeof(PrimitivePoint));
        m_IePrimitiveBuffer = DeviceBuffer(m_IePrimitiveCount * sizeof(OptixAabb));
        m_IePrimitiveInfoBuffer = DeviceBuffer(m_IePrimitiveCount * sizeof(IEPrimitiveInfo));

        m_IntersectableEntityBuffer = DeviceBuffer((m_IePrimitiveCount + m_DiffractionEdgeSegments.size() + m_Params.receivers.size()) * sizeof(IntersectableEntity));
        m_IeCountBuffer = DeviceBuffer(sizeof(uint32_t));
        m_IePointCountBuffer = DeviceBuffer(sizeof(uint32_t));
        m_IeCountBuffer.MemsetZero();
        m_IePointCountBuffer.MemsetZero();
        m_IePrimitiveCountBuffer = DeviceBuffer(sizeof(uint32_t));
        m_IePrimitiveCountBuffer.MemsetZero();

        data.iePrimitivePoints = m_IePointBuffer.DevicePointerCast<PrimitivePoint>();
        data.iePrimitives = m_IePrimitiveBuffer.DevicePointerCast<OptixAabb>();
        data.intersectableEntities = m_IntersectableEntityBuffer.DevicePointerCast<IntersectableEntity>();
        data.ieCount = m_IeCountBuffer.DevicePointerCast<uint32_t>();
        data.iePointCount = m_IePointCountBuffer.DevicePointerCast<uint32_t>();
        data.iePrimitiveCount = m_IePrimitiveCountBuffer.DevicePointerCast<uint32_t>();
        data.iePrimitiveInfos = m_IePrimitiveInfoBuffer.DevicePointerCast<IEPrimitiveInfo>();
        data.ieVoxelFactor = m_Params.ieVoxelAxisSizeFactor;
        m_VoxelPointDataBuffer = DeviceBuffer::Create(m_VoxelPointData);
        data.voxelPointData = m_VoxelPointDataBuffer.DevicePointerCast<VoxelPointData>();

        m_SubIePrimitiveCountBuffer = DeviceBuffer(sizeof(uint32_t));
        m_SubIePrimitivePointCountBuffer = DeviceBuffer(sizeof(uint32_t));
        m_SubIePrimitivePointCountBuffer.MemsetZero();
        m_SubIePrimitiveCountBuffer.MemsetZero();
        m_PerIeSubIePrimitiveCountBuffer = DeviceBuffer::Create(m_PerIeSubIePrimitiveCount);
        m_SubIePrimitivePointBuffer = DeviceBuffer(m_NumberOfSurfacePoints * sizeof(PrimitivePoint));
        m_SubIePrimitiveBuffer = DeviceBuffer(m_SubIePrimitiveCount * sizeof(OptixAabb));
        m_SubIePrimitiveInfoBuffer = DeviceBuffer(m_SubIePrimitiveCount * sizeof(IEPrimitiveInfo));
        m_SubIePrimitiveVoxelMapBuffer = DeviceBuffer(sizeof(uint32_t) * GetSubIeVoxelCount());
        m_SubIePrimitiveVoxelMapBuffer.Memset(VCT::Constants::InvalidPointIndex);
        m_SubIePrimitiveNeighborsBuffer = DeviceBuffer(sizeof(PrimitiveNeighbors) * m_SubIePrimitiveCount);
        m_SubIePrimitiveNeighborsBuffer.MemsetZero();


        data.subIe = m_Params.subIeVoxelAxisSizeFactor;
        data.subIeVoxelWorldInfo = VoxelWorldInfo(m_SceneAABB.min, data.ieVoxelWorldInfo.size / static_cast<float>(data.subIe), data.ieVoxelWorldInfo.dimensions * data.subIe);
        data.subIePrimitiveCount = m_SubIePrimitiveCountBuffer.DevicePointerCast<uint32_t>();
        data.subIePrimitivePointCount = m_SubIePrimitivePointCountBuffer.DevicePointerCast<uint32_t>();
        data.perIeSubIePrimitiveCount = m_PerIeSubIePrimitiveCountBuffer.DevicePointerCast<uint32_t>();
        data.subIePrimitivePoints = m_SubIePrimitivePointBuffer.DevicePointerCast<PrimitivePoint>();
        data.subIePrimitives = m_SubIePrimitiveBuffer.DevicePointerCast<OptixAabb>();
        data.subIePrimitiveInfos = m_SubIePrimitiveInfoBuffer.DevicePointerCast<IEPrimitiveInfo>();
        data.subIePrimitiveVoxelMap = m_SubIePrimitiveVoxelMapBuffer.DevicePointerCast<uint32_t>();
        data.subIePrimitiveNeighbors = m_SubIePrimitiveNeighborsBuffer.DevicePointerCast<PrimitiveNeighbors>();

        KernelData::Get().GetVoxelizationConstantBuffer().Upload(&data, 1);
    }

    void VoxelConeTracer::GenerateDataForRayTracing()
    {
        CreateVoxelTexture();

        uint32_t gridCount = Utils::GetLaunchCount(GetVoxelCount(), m_Params.blockSize);
        KernelData::Get().GetVoxelizePointCloudKernel().LaunchAndSynchronize(glm::uvec3(gridCount, 1, 1), glm::uvec3(m_Params.blockSize, 1, 1));
        m_IeCountBuffer.Download(&m_IeCount, 1);
        LOG("Intersectable Entity Count: %u", m_IeCount);

        gridCount = Utils::GetLaunchCount(m_IePrimitiveCount, m_Params.blockSize);
        LOG("Refine primitive count: %u %u %u", m_SubIePrimitiveCount, m_IePrimitiveCount, m_NumberOfSurfacePoints);
        KernelData::Get().GetWriteRefineAabbKernel().LaunchAndSynchronize(glm::uvec3(gridCount, 1, 1), glm::uvec3(m_Params.blockSize, 1, 1));
        gridCount = Utils::GetLaunchCount(m_SubIePrimitiveCount, m_Params.blockSize);
        KernelData::Get().GetWriteRefinePrimitiveNeighborsKernel().LaunchAndSynchronize(glm::uvec3(gridCount, 1, 1), glm::uvec3(m_Params.blockSize, 1, 1));

        m_TransmitterBuffer = DeviceBuffer::Create(m_Params.transmitters);
        m_ReceiverBuffer = DeviceBuffer::Create(m_Params.receivers);
        m_NumReceivedPathsBuffer = DeviceBuffer(sizeof(uint32_t));
        m_NumReceivedPathsBuffer.MemsetZero();

        m_CoarsePathBuffers[0] = DeviceBuffer(sizeof(TraceData) * m_Params.receivedPathBufferSize);
        m_CoarsePathBuffers[1] = DeviceBuffer(sizeof(TraceData) * m_Params.receivedPathBufferSize);
        m_ActiveBufferIndexBuffer = DeviceBuffer(sizeof(uint32_t));
        m_ActiveBufferIndexBuffer.Upload(&m_ActiveRecvBufferIndex, 1);

        m_PropPathBuffers.resize(m_Params.maximumNumberOfInteractions);
        m_PropagationStatuses.resize(m_Params.maximumNumberOfInteractions);

        std::vector<PropagationData*> propPathPtr;
        propPathPtr.reserve(m_Params.maximumNumberOfInteractions);

        m_MaxNumPropPaths.reserve(m_Params.maximumNumberOfInteractions);
        
        uint32_t propBufferSize = m_Params.propagationPathBufferSize;

        for (DeviceBuffer& propBuffer : m_PropPathBuffers)
        {
            propBuffer = DeviceBuffer(sizeof(PropagationData) * propBufferSize);
            LOG("PropBufferSize at index %u: %u", m_MaxNumPropPaths.size(), propBufferSize);
            m_MaxNumPropPaths.push_back(propBufferSize);
            propBufferSize = static_cast<uint32_t>(propBufferSize * m_Params.propagationBufferSizeIncreaseFactor);
            propPathPtr.emplace_back(propBuffer.DevicePointerCast<PropagationData>());
        }

        if (propPathPtr.size())
        {
            m_PropPathPointerBuffer = DeviceBuffer::Create(propPathPtr);
            m_MaxNumPropPathBuffer = DeviceBuffer::Create(m_MaxNumPropPaths);
        }
        if (m_DiffractionRays.size())
        {
            m_DiffractionRayBuffer = DeviceBuffer::Create(m_DiffractionRays);
            m_DiffractionRayIndexInfoBuffer = DeviceBuffer::Create(m_DiffractionRayIndexInfos);

            m_DiffractionEdgeBuffer = DeviceBuffer::Create(m_DiffractionEdges);
            m_DiffractionEdgeSegmentBuffer = DeviceBuffer::Create(m_DiffractionEdgeSegments);
        }

        m_PathProcessingDataBuffer = DeviceBuffer(sizeof(PathProcessingData));
        m_PathProcessingDataBuffer.MemsetZero();

        m_TransmitIndexProcessedBuffer = DeviceBuffer(sizeof(uint8_t) * m_IeCount);
        m_VCTStatusBuffer = DeviceBuffer(sizeof(Status));
        m_DepthLevelBuffer = DeviceBuffer(sizeof(int32_t));
    }

    void VoxelConeTracer::CreateVoxelTexture()
    {
        cudaArray* arr;
        cudaExtent extent = make_cudaExtent(m_VoxelDimensions.x, m_VoxelDimensions.y, m_VoxelDimensions.z);
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uint2>();
        CUDA_CHECK(cudaMalloc3DArray(&arr, &desc, extent, 0));

        uint32_t gridCount = Utils::GetLaunchCount(GetVoxelCount(), m_Params.blockSize);
        KernelData::Get().GetFillTextureDataKernel().LaunchAndSynchronize(glm::vec3(gridCount, 1, 1), glm::vec3(m_Params.blockSize, 1, 1));

        cudaMemcpy3DParms copyParams{};
        copyParams.srcPtr = make_cudaPitchedPtr(m_VoxelTextureDataBuffer.DevicePointerCast<void>(), extent.width * sizeof(uint2), extent.width, extent.height);
        copyParams.dstArray = arr;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        CUDA_CHECK(cudaMemcpy3D(&copyParams));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaResourceDesc resourceDesc{};
        resourceDesc.resType = cudaResourceTypeArray;
        resourceDesc.res.array.array = arr;

        cudaTextureDesc textureDesc{};
        textureDesc.addressMode[0] = cudaAddressModeBorder;
        textureDesc.addressMode[1] = cudaAddressModeBorder;
        textureDesc.addressMode[2] = cudaAddressModeBorder;
        textureDesc.filterMode = cudaFilterModePoint;
        textureDesc.readMode = cudaReadModeElementType;
        textureDesc.sRGB = false;
        uint32_t borderColor = Constants::InvalidPointIndex;
        textureDesc.borderColor[0] = reinterpret_cast<float&>(borderColor);
        textureDesc.borderColor[1] = reinterpret_cast<float&>(borderColor);
        textureDesc.normalizedCoords = false;
        textureDesc.maxAnisotropy = 0;
        textureDesc.mipmapFilterMode = cudaFilterModePoint;
        textureDesc.mipmapLevelBias = 0.0f;
        textureDesc.minMipmapLevelClamp = 0.0f;
        textureDesc.maxMipmapLevelClamp = 0.0f;

        cudaResourceViewDesc viewDesc{};
        viewDesc.format = cudaResViewFormatUnsignedInt2;
        viewDesc.width = m_VoxelDimensions.x;
        viewDesc.height = m_VoxelDimensions.y;
        viewDesc.depth = m_VoxelDimensions.z;
        viewDesc.firstMipmapLevel = 0;
        viewDesc.lastMipmapLevel = 0;
        viewDesc.firstLayer = 0;
        viewDesc.lastLayer = 0;

        CUDA_CHECK(cudaCreateTextureObject(&m_VoxelTexture, &resourceDesc, &textureDesc, &viewDesc));
    }

    void VoxelConeTracer::PrepareTrace(uint32_t transmitterID)
    {
        m_VCTData.currentTransmitterID = transmitterID;
        m_TransmitIndexProcessedBuffer.MemsetZero();
        m_VCTDataBuffer.Upload(&m_VCTData, 1);
        
        Status status = Status::Finished;
        m_TransmitStatus = Status::ProcessingRequired;
        m_VCTStatusBuffer.Upload(&status, 1);
     
        PathProcessingData ppData{};
        ppData.numPaths = 0;
        ppData.numPathsToProcess = m_IeCount;
        ppData.nextNumPathsToProcess = 0;
        m_PathProcessingDataBuffer.Upload(&ppData, 1);
        m_DepthLevel = -1;
    }

    void VoxelConeTracer::IncreaseDepth()
    {
        if (++m_DepthLevel < static_cast<int32_t>(m_Params.maximumNumberOfInteractions))
        {
            PathProcessingData ppData{};
            m_PathProcessingDataBuffer.Download(&ppData, 1);
            ppData.numPathsToProcess = ppData.nextNumPathsToProcess;
            ppData.numPaths = 0;
            ppData.nextNumPathsToProcess = 0;
            m_PathProcessingDataBuffer.Upload(&ppData, 1);
            m_PropagationStatuses[m_DepthLevel] = { Status::ProcessingRequired, ppData.numPathsToProcess };
            Status resetStatus = Status::Finished;
            m_VCTStatusBuffer.Upload(&resetStatus, 1);
            m_DepthLevelBuffer.Upload(&m_DepthLevel, 1);
        }
    }
   
    void VoxelConeTracer::DecreaseDepth()
    {
        if (--m_DepthLevel >= 0)
        {
            m_DepthLevelBuffer.Upload(&m_DepthLevel, 1);
            PropagationStatus& propStatus = m_PropagationStatuses[m_DepthLevel];
            PathProcessingData ppData{};
            ppData.numPathsToProcess = propStatus.launchCount;
            m_PathProcessingDataBuffer.Upload(&ppData, 1);
            Status resetStatus = Status::Finished;
            m_VCTStatusBuffer.Upload(&resetStatus, 1);
        }
    }

    void VoxelConeTracer::Transmit()
    {   
        if (m_IeCount > 0 && m_TransmitStatus == Status::ProcessingRequired)
        {
            PROFILE_SCOPE();
            LOG("Transmit Launch Count: %u", m_IeCount);
            KernelData::Get().GetTransmitPipeline().LaunchAndSynchronize(m_VCTDataBuffer, glm::uvec3(m_IeCount, 1, 1));
            m_VCTStatusBuffer.Download(&m_TransmitStatus, 1);
            IncreaseDepth();
        }
        else
            DecreaseDepth();
    }

    void VoxelConeTracer::Propagate()
    {
        if (m_DepthLevel < static_cast<int32_t>(m_Params.maximumNumberOfInteractions) && m_PropagationStatuses[m_DepthLevel].ProcessingRequired())
        {
            PROFILE_SCOPE();            
            PropagationStatus& propStatus = m_PropagationStatuses[m_DepthLevel];
            LOG("Propagate depthLevel: %i, launchCount %u", m_DepthLevel, propStatus.launchCount);
            KernelData::Get().GetPropagationPipeline().LaunchAndSynchronize(m_VCTDataBuffer, glm::uvec3(propStatus.launchCount, 1, 1));
            m_VCTStatusBuffer.Download(&propStatus.status, 1);
            uint32_t numPaths = 0;
            m_NumReceivedPathsBuffer.Download(&numPaths, 1);
            if (numPaths >= m_Params.receivedPathBufferSize)
            {
                if (m_TransferStatus.valid())
                    m_TransferStatus.wait();

                m_NumReceivedPathsBuffer.MemsetZero();
                uint32_t bufferIdx = m_ActiveRecvBufferIndex;
                m_ActiveRecvBufferIndex = m_ActiveRecvBufferIndex ^ 1u;
                m_TransferStatus = std::async(std::launch::async, &VoxelConeTracer::RetrievePaths, this, &m_CoarsePathBuffers[bufferIdx], glm::min(numPaths, m_Params.receivedPathBufferSize));
                m_ActiveBufferIndexBuffer.Upload(&m_ActiveRecvBufferIndex, 1);
            }
            IncreaseDepth();
        }
        else
            DecreaseDepth();
    }

    void VoxelConeTracer::TraceTransmitter(uint32_t transmitterID)
    {
        PrepareTrace(transmitterID);
        while (m_DepthLevel >= -1)
        {
            (m_DepthLevel == -1) ? Transmit() : Propagate();
        }
        uint32_t numPaths = 0;
        m_NumReceivedPathsBuffer.Download(&numPaths, 1);
        if (numPaths)
        {
            if (m_TransferStatus.valid())
                m_TransferStatus.wait();

            m_NumReceivedPathsBuffer.MemsetZero();
            uint32_t bufferIdx = m_ActiveRecvBufferIndex;
            m_ActiveRecvBufferIndex = m_ActiveRecvBufferIndex ^ 1u;
            RetrievePaths(&m_CoarsePathBuffers[bufferIdx], glm::min(numPaths, m_Params.receivedPathBufferSize));
            m_ActiveBufferIndexBuffer.Upload(&m_ActiveRecvBufferIndex, 1);

            uint32_t totalPaths = 0;
            for (uint32_t rxID = 0; rxID < m_Params.receivers.size(); ++rxID)
            {
                auto paths = m_CoarsePathStorage.GetPaths(transmitterID, rxID);
                totalPaths += paths ? static_cast<uint32_t>(paths->size()) : 0u;
            }

            LOG("Coarse paths for TX: %u\n", totalPaths);
        }
    }

    void VoxelConeTracer::CalculateDiffractionRays()
    {
        PROFILE_SCOPE();
        float length = glm::length(m_SceneAABB.max - m_SceneAABB.min);
        constexpr float voxelSampleRadiusForDiffractions = 1.0f;
        float radius = m_Params.voxelSize * voxelSampleRadiusForDiffractions;
        float radiusSq = radius * radius;
        float separationMaxRadius = glm::sqrt(radiusSq + radiusSq);
        float radHalfDiffAngle = glm::atan(separationMaxRadius / length);
        m_MaxDiffuseAngle = static_cast<float>(radHalfDiffAngle);
        m_DiffuseAngleSin = glm::sin(m_MaxDiffuseAngle);
        m_DiffuseAngleCos = glm::cos(m_MaxDiffuseAngle);

        float radDiffAngle = m_MaxDiffuseAngle * 2;
        constexpr float pi2 = glm::pi<float>() * 2;
        uint32_t numDiffRays = static_cast<uint32_t>(pi2 / radDiffAngle);
        radDiffAngle = pi2 / numDiffRays;

        float percent = static_cast<float>(numDiffRays) / static_cast<float>(Constants::UnitCircleDiscretizationCount);
        constexpr float factor = 1.0f / static_cast<float>(Constants::UnitCircleDiscretizationCount);

        uint32_t infosRayFirstIndex = 0;
        m_DiffractionRayIndexInfos.reserve(static_cast<size_t>(Constants::UnitCircleDiscretizationCount + 1) * m_DiffractionEdges.size());
        m_DiffractionRays.reserve(static_cast<size_t>(Constants::UnitCircleDiscretizationCount + 1) * m_DiffractionEdges.size() * numDiffRays);

        for (DiffractionEdge& edge : m_DiffractionEdges)
        {
            edge.firstInfoIndex = static_cast<uint32_t>(m_DiffractionRayIndexInfos.size());
            float startAngle = FindDiffractionAngleRadsFromNormals(edge.localSurfaceDir2D0, edge.localSurfaceDir2D1);
            float surfaceAngleCos = glm::dot(edge.localSurfaceDir2D0, edge.localSurfaceDir2D1);
            float surfaceAngleRads = std::acosf(surfaceAngleCos);
            float diffRayAreaRads = pi2 - surfaceAngleRads;

            for (size_t diffractionIndex = 0; diffractionIndex <= Constants::UnitCircleDiscretizationCount; ++diffractionIndex)
            {
                float cAngle = radDiffAngle / ((diffractionIndex + 1) * factor);
                float halfAngle = cAngle / 2;
                uint32_t numRays = static_cast<uint32_t>(std::ceil(diffRayAreaRads / cAngle));
                cAngle = diffRayAreaRads / numRays;
                float angle = startAngle + halfAngle;

                m_DiffractionRayIndexInfos.push_back({ infosRayFirstIndex, numRays });
                infosRayFirstIndex += numRays;

                for (uint32_t rayIndex = 0; rayIndex < numRays; ++rayIndex)
                {
                    DiffractionRay ray{};
                    ray.direction = glm::vec2(std::sinf(angle), std::cosf(angle));
                    ray.planeDirections[0] = glm::vec2(std::sinf(angle + halfAngle), std::cosf(angle + halfAngle));
                    ray.planeDirections[1] = glm::vec2(std::sinf(angle - halfAngle), std::cosf(angle - halfAngle));
                    m_DiffractionRays.push_back(ray);
                    angle += cAngle;
                }
            }
        }
    }

    VCTData VoxelConeTracer::CreateVCTData() const
    {
        VCTData vctData{};
        vctData.sceneData.intersectableEntities = m_IntersectableEntityBuffer.DevicePointerCast<IntersectableEntity>();
        vctData.sceneData.transmitters = m_TransmitterBuffer.DevicePointerCast<Transmitter>();
        vctData.sceneData.receivers = m_ReceiverBuffer.DevicePointerCast<Receiver>();
        
        vctData.sceneData.rtParams.asHandle = m_AccelerationStructure.GetRawHandle();
        vctData.sceneData.rtParams.primitives = m_SubIePrimitiveBuffer.DevicePointerCast<OptixAabb>();
        vctData.sceneData.rtParams.primitivePoints = m_SubIePrimitivePointBuffer.DevicePointerCast<PrimitivePoint>();
        vctData.sceneData.rtParams.primitiveInfos = m_SubIePrimitiveInfoBuffer.DevicePointerCast<IEPrimitiveInfo>();
        vctData.sceneData.rtParams.sampleDistance = glm::length(glm::vec3(GetIeVoxelSize())) / m_Params.subIeVoxelAxisSizeFactor;
        vctData.sceneData.rtParams.sampleRadius = m_Params.sampleRadiusCoarse;
        vctData.sceneData.rtParams.traceDistanceBias = vctData.sceneData.rtParams.sampleDistance * 0.5f;
        vctData.sceneData.rtParams.varianceFactor = m_Params.varianceFactorCoarse;
        vctData.sceneData.rtParams.sdfThreshold = m_Params.sdfThresholdCoarse;

        vctData.sceneData.refineRtParams.asHandle = m_AccelerationStructure.GetRawHandle();
        vctData.sceneData.refineRtParams.primitives = m_SubIePrimitiveBuffer.DevicePointerCast<OptixAabb>();
        vctData.sceneData.refineRtParams.primitivePoints = m_SubIePrimitivePointBuffer.DevicePointerCast<PrimitivePoint>();
        vctData.sceneData.refineRtParams.primitiveInfos = m_SubIePrimitiveInfoBuffer.DevicePointerCast<IEPrimitiveInfo>();
        vctData.sceneData.refineRtParams.sampleDistance = glm::length(glm::vec3(GetIeVoxelSize())) / m_Params.subIeVoxelAxisSizeFactor;
        vctData.sceneData.refineRtParams.sampleRadius = m_Params.sampleRadiusRefine;
        vctData.sceneData.refineRtParams.traceDistanceBias = vctData.sceneData.refineRtParams.sampleDistance * 0.5f;
        vctData.sceneData.refineRtParams.varianceFactor = m_Params.varianceFactorRefine;
        vctData.sceneData.refineRtParams.sdfThreshold = m_Params.sdfThresholdRefine;
        vctData.subIePrimitiveNeighbors = m_SubIePrimitiveNeighborsBuffer.DevicePointerCast<PrimitiveNeighbors>();

        vctData.coneTracingData.voxelTexture = m_VoxelTexture;
        vctData.coneTracingData.voxelInfos = m_VoxelInfoBuffer.DevicePointerCast<VoxelInfo>();
        vctData.coneTracingData.voxelWorldInfo = VoxelWorldInfo(m_SceneAABB.min, m_Params.voxelSize, m_VoxelDimensions);
        vctData.coneTracingData.maximumNumberOfInteractions = m_Params.maximumNumberOfInteractions;
        vctData.coneTracingData.maximumNumberOfDiffractions = m_Params.maximumNumberOfDiffractions;
        vctData.coneTracingData.ieBoundingSphereRadius = glm::length(glm::vec3(1.0f / m_Params.ieVoxelAxisSizeFactor));
        vctData.coneTracingData.diffractionEdges = m_DiffractionEdgeBuffer.DevicePointerCast<DiffractionEdge>();
        vctData.coneTracingData.diffractionEdgeSegments = m_DiffractionEdgeSegmentBuffer.DevicePointerCast<DiffractionEdgeSegment>();
        vctData.coneTracingData.diffractionRays = m_DiffractionRayBuffer.DevicePointerCast<DiffractionRay>();
        vctData.coneTracingData.diffractionIndexInfos = m_DiffractionRayIndexInfoBuffer.DevicePointerCast<IndexInfo>();
        vctData.coneTracingData.sinDiffuseAngle = m_DiffuseAngleSin;
        vctData.coneTracingData.cosDiffuseAngle = m_DiffuseAngleCos;
        
        vctData.coneTracingData.reflSinDiffuseAngle = m_Params.useConeReflections ? m_DiffuseAngleSin : 0.0f;
        vctData.coneTracingData.reflCosDiffuseAngle = m_Params.useConeReflections ? m_DiffuseAngleCos : 1.0f;
        vctData.coneTracingData.depthLevel = m_DepthLevelBuffer.DevicePointerCast<int32_t>();

        vctData.pathData.maxNumReceivedPaths = m_Params.receivedPathBufferSize;
        vctData.pathData.numReceivedPaths = m_NumReceivedPathsBuffer.DevicePointerCast<uint32_t>();
        vctData.pathData.propPaths = m_PropPathPointerBuffer.DevicePointerCast<PropagationData*>();
        vctData.pathData.maxNumPropPaths = m_MaxNumPropPathBuffer.DevicePointerCast<uint32_t>();
        vctData.pathData.pathProcessingData = m_PathProcessingDataBuffer.DevicePointerCast<PathProcessingData>();
        vctData.pathData.coarsePaths[0] = m_CoarsePathBuffers[0].DevicePointerCast<TraceData>();
        vctData.pathData.coarsePaths[1] = m_CoarsePathBuffers[1].DevicePointerCast<TraceData>();
        vctData.pathData.activeBufferIndex = m_ActiveBufferIndexBuffer.DevicePointerCast<uint32_t>();

        vctData.transmitIndexProcessed = m_TransmitIndexProcessedBuffer.DevicePointerCast<uint8_t>();
        vctData.status = m_VCTStatusBuffer.DevicePointerCast<Status>();

        vctData.refineParams = m_Params.refineParams;

        return vctData;
    }

    void VoxelConeTracer::RetrievePaths(DeviceBuffer* deviceBuffer, uint32_t numPaths)
    {
        cudaStream_t stream{};
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        deviceBuffer->DownloadAsync(stream, m_TransferHostBuffer.get(), numPaths);
        m_CoarsePathStorage.AddPaths(m_TransferHostBuffer.get(), numPaths, m_UseLabelHashing);
        LOG("Retrieved %u coarse paths from device.", numPaths);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void VoxelConeTracer::PostProcess(uint32_t txID, uint32_t rxID)
    {
        PROFILE_SCOPE();
        auto* p = m_RefinedPathStorage.GetPaths(0, 0);
        m_RefinedPathStorage.TryRemoveDuplicates(0, 0, m_Params.transmitters[0], m_Params.receivers[0], m_Channel.waveLength);
        LOG("Number of refined paths after duplicate removal: %u", (p ? p->size() : 0));
    }
}