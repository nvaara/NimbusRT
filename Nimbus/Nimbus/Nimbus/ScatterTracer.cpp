#include "ScatterTracer.hpp"
#include "Nimbus/Utils.hpp"
#include "KernelData.hpp"
#include "Logger.hpp"
#include <fstream>
#include <memory>

namespace Nimbus
{
	ScatterTracer::ScatterTracer()
        : m_Environment(nullptr)
        , m_STRTData({})
        , m_PropagationPathCount(0u)
        , m_IeCount(0u)
        , m_TxCount(0u)
        , m_RxCount(0u)
        , m_MaxNumIa(0u)
        , m_Los(false)
        , m_Reflection(false)
        , m_Scattering(false)
        , m_Diffraction(false)
        , m_Ris(false)
	{

	}

    bool ScatterTracer::Prepare(Environment& env,
                                const ScatterTracingParams& params,
                                const glm::vec3* txs,
                                uint32_t txCount,
                                const glm::vec3* rxs,
                                uint32_t rxCount,
                                const RisData& risData)
    {
        m_Environment = &env;
        if (txCount == 0 || rxCount == 0)
        {
            LOG("Invalid number of antennas");
            return false;
        }

        env.InitRisGasData(risData);

        m_IeCount = env.GetRtPointCount();
        m_TxCount = txCount;
        m_RxCount = rxCount;
        m_MaxNumIa = params.maxNumInteractions;
        m_Los = params.los;
        m_Reflection = params.reflection;
        m_Scattering = params.scattering;
        m_Diffraction = params.diffraction && env.HasEdges();
        m_Ris = params.ris && env.GetRisPointCount() > 0u;

        m_TransmitterBuffer = DeviceBuffer(sizeof(glm::vec3) * txCount);
        m_TransmitterBuffer.Upload(txs, txCount);
        m_ReceiverBuffer = DeviceBuffer(sizeof(glm::vec3) * rxCount);
        m_ReceiverBuffer.Upload(rxs, rxCount);

        m_ScattererVisibleBuffer = DeviceBuffer(m_IeCount / 8u + sizeof(uint32_t));
        m_RxVisibleBuffer = DeviceBuffer(rxCount * m_IeCount / 8u + sizeof(uint32_t));
        m_ScattererVisibleBuffer.MemsetZero();
        m_RxVisibleBuffer.MemsetZero();
        m_STRTDataBuffer = DeviceBuffer(sizeof(STRTData));

        m_STRTData.transmitters = m_TransmitterBuffer.DevicePointerCast<glm::vec3>();
        m_STRTData.receivers = m_ReceiverBuffer.DevicePointerCast<glm::vec3>();

        m_STRTData.rtParams.env = env.GetGpuEnvironmentData();
        m_STRTData.rtParams.rayBias = params.rayBias;

        m_STRTData.refineParams.maxNumIterations = params.numRefineIterations;
        m_STRTData.refineParams.maxCorrectionIterations = params.refineMaxCorrectionIterations;
        m_STRTData.refineParams.delta = params.refineDelta;
        m_STRTData.refineParams.beta = params.refineBeta;
        m_STRTData.refineParams.alpha = params.refineAlpha;
        m_STRTData.refineParams.angleThreshold = glm::cos(glm::radians(params.refineAngleDegreesThreshold));
        m_STRTData.refineParams.distanceThreshold = params.refineDistanceThreshold;

        m_STRTData.scattererVisible = m_ScattererVisibleBuffer.DevicePointerCast<uint32_t>();
        m_STRTData.rxVisible = m_RxVisibleBuffer.DevicePointerCast<uint32_t>();
        m_STRTData.numRx = static_cast<uint32_t>(rxCount);

        uint32_t pathCount = glm::max(glm::max(glm::max(m_IeCount, m_Environment->GetEdgeCount()), m_Environment->GetRisPointCount()), m_RxCount);

        m_PathProcessedBuffer = DeviceBuffer(rxCount * pathCount / 8u + sizeof(uint32_t));
        m_PathProcessedBuffer.MemsetZero();
        
        m_PathCountBuffer = DeviceBuffer(sizeof(uint32_t));
        m_PathCountBuffer.MemsetZero();
        
        m_ReceivedPathCountBuffer = DeviceBuffer(sizeof(uint32_t));
        m_ReceivedPathCountBuffer.MemsetZero();
        
        m_ReceivedPathData.Resize(pathCount, params.maxNumInteractions);
        m_PropagationPathData.Resize(pathCount, params.maxNumInteractions);

        m_PathStorage = std::make_unique<PathStorage>(params.maxNumInteractions, txs, txCount, rxs, rxCount);
        m_PathInfos.resize(pathCount);
        m_Interactions.resize(pathCount * params.maxNumInteractions);
        m_Normals.resize(pathCount * params.maxNumInteractions);
        m_Labels.resize(pathCount * params.maxNumInteractions);
        m_Materials.resize(pathCount * params.maxNumInteractions);

        m_STRTData.propagationPathData.interactions = m_PropagationPathData.interactionBuffer.DevicePointerCast<glm::vec3>();
        m_STRTData.propagationPathData.normals = m_PropagationPathData.normalBuffer.DevicePointerCast<glm::vec3>();
        m_STRTData.propagationPathData.labels = m_PropagationPathData.labelBuffer.DevicePointerCast<uint32_t>();
        m_STRTData.propagationPathData.materials = m_PropagationPathData.materialBuffer.DevicePointerCast<uint32_t>();
        m_STRTData.propagationPathData.pathInfos = m_PropagationPathData.pathInfoBuffer.DevicePointerCast<PathInfoST>();

        m_STRTData.receivedPathData.interactions = m_ReceivedPathData.interactionBuffer.DevicePointerCast<glm::vec3>();
        m_STRTData.receivedPathData.normals = m_ReceivedPathData.normalBuffer.DevicePointerCast<glm::vec3>();
        m_STRTData.receivedPathData.labels = m_ReceivedPathData.labelBuffer.DevicePointerCast<uint32_t>();
        m_STRTData.receivedPathData.materials = m_ReceivedPathData.materialBuffer.DevicePointerCast<uint32_t>();
        m_STRTData.receivedPathData.pathInfos = m_ReceivedPathData.pathInfoBuffer.DevicePointerCast<PathInfo>();

        m_STRTData.maxNumIa = params.maxNumInteractions;

        m_STRTData.numReceivedPathsMax = pathCount;
        m_STRTData.pathsProcessed = m_PathProcessedBuffer.DevicePointerCast<uint32_t>();
        m_STRTData.pathCount = m_PathCountBuffer.DevicePointerCast<uint32_t>();
        m_STRTData.receivedPathCount = m_ReceivedPathCountBuffer.DevicePointerCast<uint32_t>();
        const Aabb& aabb = env.GetAabb();
        m_STRTData.rtParams.rayMaxLength = glm::length(aabb.max - aabb.min);
        return true;
    }

    bool ScatterTracer::CreateCoverageMapInfo(Environment& env, const glm::vec3& tx, float size, float height, CoverageMapInfo& result, std::vector<glm::vec3>& receivers, const RisData& risData)
    {
        glm::uvec3 voxelDimensions = glm::uvec3(glm::ceil((env.GetAabb().max - env.GetAabb().min) / size));
        if (size <= 0.0f)
        {
            LOG("Failed to create coverage grid with size: %f", size);
            return false;
        }

        uint32_t numGridPoints = voxelDimensions.x * voxelDimensions.y;
        DeviceBuffer rxBuffer = DeviceBuffer(sizeof(glm::vec3) * numGridPoints);
        DeviceBuffer rx2DBuffer = DeviceBuffer(sizeof(glm::uvec2) * numGridPoints);
        DeviceBuffer numRxBuffer = DeviceBuffer(sizeof(uint32_t));
        DeviceBuffer cellProcessedBuffer = DeviceBuffer(numGridPoints / 8u + sizeof(uint32_t));
        numRxBuffer.MemsetZero();
        cellProcessedBuffer.MemsetZero();

        EnvironmentData envData = env.GetGpuEnvironmentData();

        CoverageData data{};
        data.cellProcessed = cellProcessedBuffer.DevicePointerCast<uint32_t>();
        data.coverageWorldInfo = Nimbus::VoxelWorldInfo(envData.vwInfo.worldOrigin, size, voxelDimensions);
        data.outReceivers = rxBuffer.DevicePointerCast<glm::vec3>();
        data.outNumRx = numRxBuffer.DevicePointerCast<uint32_t>();
        data.outRx2D = rx2DBuffer.DevicePointerCast<glm::uvec2>();
        data.rtPoints = envData.rtPoints;
        data.numPoints = env.GetRtPointCount();
        data.height = height;

        KernelData::Get().GetStCoverageConstantBuffer().Upload(&data, 1);
        constexpr uint32_t blockSize = 256u;
        uint32_t gridCount = Utils::GetLaunchCount(data.numPoints, blockSize);
        KernelData::Get().GetStCoveragePointsKernel().LaunchAndSynchronize(glm::uvec3(gridCount, 1u, 1u), glm::uvec3(blockSize, 1u, 1u));

        uint32_t numRx = 0;
        numRxBuffer.Download(&numRx, 1);
        if (numRx == 0)
        {
            LOG("No grid points in point cloud found.");
            return false;
        }

        receivers.resize(numRx);
        std::vector<glm::uvec2> rxCoords2D(numRx);
        rxBuffer.Download(receivers.data(), receivers.size());
        rx2DBuffer.Download(rxCoords2D.data(), rxCoords2D.size());
        result = CoverageMapInfo(glm::uvec2(voxelDimensions.x, voxelDimensions.y), std::move(rxCoords2D), size, height);
        return true;
    }

    std::unique_ptr<PathStorage> ScatterTracer::Trace()
    {
        ComputeVisibility();
        for (uint32_t txID = 0u; txID < static_cast<uint32_t>(m_TxCount); ++txID)
        {
            Transmit(txID);
            if (m_PropagationPathCount > 0u)
            {
                if (m_MaxNumIa > 0u)
                    Refine();

                for (uint32_t i = 1u; i < m_MaxNumIa; ++i)
                {
                    Propagate();
                    Refine();
                }
            }
        }
        return std::move(m_PathStorage);
    }

    void ScatterTracer::ComputeVisibility()
    {
        m_STRTDataBuffer.Upload(&m_STRTData, 1);
        m_Environment->ComputeVisibility(m_STRTDataBuffer, glm::uvec3(m_IeCount, static_cast<uint32_t>(m_RxCount), 1u));
    }

    void ScatterTracer::DetermineLOSPaths()
    {
        if (m_Los)
        {
          m_Environment->DetermineLosPaths(m_STRTDataBuffer, glm::uvec3(static_cast<uint32_t>(m_RxCount), 1u, 1u));
          RetrieveReceivedPaths();
        }
    }

    void ScatterTracer::Transmit(uint32_t txID)
    {
        m_STRTData.currentTxID = txID;
        m_STRTDataBuffer.Upload(&m_STRTData, 1u);
        DetermineLOSPaths();
        if (m_MaxNumIa > 0)
        {
            ComputeDiffractionPaths();
            ComputeRisPaths();
            m_Environment->Transmit(m_STRTDataBuffer, glm::uvec3(m_IeCount, 1u, 1u));
            m_PathCountBuffer.Download(&m_PropagationPathCount, 1u);
        }
    }

    void ScatterTracer::Propagate()
    {
        m_Environment->Propagate(m_STRTDataBuffer, glm::uvec3(m_PropagationPathCount, 1u, 1u));
    }

    void ScatterTracer::Refine()
    {
        if (m_Reflection)
        {
            m_PathProcessedBuffer.MemsetZero();
            do
            {
                m_Environment->RefineSpecular(m_STRTDataBuffer, glm::uvec3(m_PropagationPathCount, m_RxCount, 1u));
            } while (RetrieveReceivedPaths());
        }

        if (m_Scattering)
        { 
            m_PathProcessedBuffer.MemsetZero();
            do
            {
                m_Environment->RefineScatterer(m_STRTDataBuffer, glm::uvec3(m_PropagationPathCount, m_RxCount, 1u));
            } while(RetrieveReceivedPaths());
        }
    }

    bool ScatterTracer::RetrieveReceivedPaths()
    {
        uint32_t recvPathCount = 0;
        m_ReceivedPathCountBuffer.Download(&recvPathCount, 1);
        if (recvPathCount)
        {
            uint32_t recvClamped = glm::min(recvPathCount, m_STRTData.numReceivedPathsMax);
            m_ReceivedPathData.pathInfoBuffer.Download(m_PathInfos.data(), recvClamped);
            m_ReceivedPathData.interactionBuffer.Download(m_Interactions.data(), recvClamped * m_MaxNumIa);
            m_ReceivedPathData.normalBuffer.Download(m_Normals.data(), recvClamped * m_MaxNumIa);
            m_ReceivedPathData.labelBuffer.Download(m_Labels.data(), recvClamped * m_MaxNumIa);
            m_ReceivedPathData.materialBuffer.Download(m_Materials.data(), recvClamped * m_MaxNumIa);
            m_PathStorage->AddPaths(recvClamped, m_PathInfos, m_Interactions, m_Normals, m_Labels, m_Materials);
            m_ReceivedPathCountBuffer.MemsetZero();
        }
        return recvPathCount > m_STRTData.numReceivedPathsMax;
    }

    void ScatterTracer::ComputeDiffractionPaths()
    {
        if (m_Diffraction)
        {
            glm::vec3 dims = glm::uvec3(m_Environment->GetEdgeCount(), m_RxCount, 1u);
            m_PathProcessedBuffer.MemsetZero();
            do
            {
                m_Environment->RefineDiffraction(m_STRTDataBuffer, dims);
            } while (RetrieveReceivedPaths());
        }
    }

    void ScatterTracer::ComputeRisPaths()
    {
        if (m_Ris)
        {
            glm::vec3 dims = glm::uvec3(m_Environment->GetRisPointCount(), m_RxCount, 1u);
            m_PathProcessedBuffer.MemsetZero();
            do
            {
                m_Environment->ComputeRISPaths(m_STRTDataBuffer, dims);
            } while (RetrieveReceivedPaths());
        }
    }
}