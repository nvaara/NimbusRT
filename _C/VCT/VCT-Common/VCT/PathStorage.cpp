#include "PathStorage.hpp"
#include "Common.hpp"
#include <algorithm>
#include <array>

namespace VCT
{
    namespace
    {
        size_t TxRxHash(const TraceData& traceData)
        {
            return CalculateHash(traceData.transmitterID, traceData.receiverID);
        }

        size_t GetPathHash(const TraceData& traceData)
        {
            size_t hash = CalculateHash(traceData.transmitterID, traceData.receiverID);
            for (uint32_t i = 0; i < traceData.numInteractions; ++i)
                CombineHash(hash, traceData.interactions[i].label, static_cast<uint32_t>(traceData.interactions[i].type));
            return hash;
        }
    }

    PathStorage::PathStorage(uint32_t pathsPerHash)
        : m_PathsPerHash(pathsPerHash)
    {
    }

    void PathStorage::AddPaths(const std::vector<TraceData>& traceDatas, bool useHash)
    {
        for (auto& traceData : traceDatas)
            AddPath(traceData, useHash);
    }

    void PathStorage::AddPaths(const TraceData* traceDatas, uint32_t numPaths, bool useHash)
    {
        for (uint32_t i = 0; i < numPaths; ++i)
            AddPath(traceDatas[i], useHash);
    }

    void PathStorage::AddPath(const TraceData& traceData, bool useHash)
    {
        std::vector<TraceData>& txRxPaths = m_PathMap[TxRxHash(traceData)];
        if (useHash)
        {
             PathReference& pathRef = m_PathHashReferenceMap[GetPathHash(traceData)];
             if (pathRef.references.size() == 0)
                 pathRef.references.reserve(m_PathsPerHash);

             if (pathRef.references.size() == m_PathsPerHash)
             {
                 if (traceData.timeDelay > pathRef.maxTimeDelay)
                     return;
             
                 txRxPaths.at(pathRef.references[pathRef.longestPathIndex]) = traceData;
                 pathRef.maxTimeDelay = traceData.timeDelay;
                 for (uint32_t i = 0; i < pathRef.references.size(); ++i)
                 {
                     if (txRxPaths[pathRef.references[i]].timeDelay > pathRef.maxTimeDelay)
                     {
                         pathRef.longestPathIndex = i;
                         pathRef.maxTimeDelay = txRxPaths[pathRef.references[i]].timeDelay;
                     }
                 }
                 return;
             }
             else
             {
                 if (traceData.timeDelay > pathRef.maxTimeDelay)
                 {
                     pathRef.maxTimeDelay = traceData.timeDelay;
                     pathRef.longestPathIndex = static_cast<uint32_t>(pathRef.references.size());
                 }
                 pathRef.references.push_back(static_cast<uint32_t>(txRxPaths.size()));
             }
        }
        txRxPaths.push_back(traceData);
    }

    struct FresnelZone
    {
        FresnelZone() : point(glm::vec3(0.0f)), radiusSq(0.0f), iaType(VCT::InteractionType::Reflection), dstDir(0.0f), srcDir(0.0f) {}
        FresnelZone(const VCT::Interaction& ia, const glm::vec3& src, const glm::vec3& dst, float waveLength)
            : point(ia.position)
            , radiusSq([&]() -> float {float d1 = glm::length(src - ia.position); float d2 = glm::length(dst - ia.position); return (d1 * d2) / (d1 + d2) * waveLength; }())
            , iaType(ia.type)
            , srcDir(glm::normalize(ia.position - src))
            , dstDir(glm::normalize(dst - ia.position))
        {

        }

        bool InZone(const VCT::Interaction& ia, const glm::vec3& src, const glm::vec3& dst) const
        {
            glm::vec3 diff = ia.position - point;
            return iaType == ia.type  && glm::dot(glm::normalize(ia.position - src), srcDir) > 0.99f && glm::dot(glm::normalize(dst - ia.position), dstDir) > 0.99f && glm::dot(diff, diff) <= radiusSq;
        }

        glm::vec3 point;
        float radiusSq;
        VCT::InteractionType iaType;
        glm::vec3 srcDir;
        glm::vec3 dstDir;
    };

    struct PathFresnelZones
    {
        PathFresnelZones(const TraceData& traceData, const Transmitter& transmitter, const Receiver& receiver, float waveLength)
            : numZones(traceData.numInteractions)
        {
            if (numZones > 1)
            {
                zones[0] = FresnelZone(traceData.interactions[0], transmitter.position, traceData.interactions[1].position, waveLength);

                for (int32_t i = 1; i < numZones - 1; ++i)
                    zones[i] = FresnelZone(traceData.interactions[i], traceData.interactions[i - 1].position, traceData.interactions[i + 1].position, waveLength);

                zones[numZones - 1] = FresnelZone(traceData.interactions[numZones - 1], traceData.interactions[numZones - 2].position, receiver.position, waveLength);
            }
            else
                zones[0] = FresnelZone(traceData.interactions[0], transmitter.position, receiver.position, waveLength);
        }

        bool IsSharedZone(const TraceData& traceData, const glm::vec3& tx, const glm::vec3& rx) const
        { 
            if (traceData.numInteractions != numZones)
                return false;
            
            for (int32_t i = 0; i < numZones; ++i)
            {
                glm::vec3 src = (i == 0) ? tx : traceData.interactions[i - 1].position;
                glm::vec3 dst = (i == numZones - 1) ? rx : traceData.interactions[i + 1].position;
                if (!zones[i].InZone(traceData.interactions[i], src, dst))
                    return false;
            }
            return true;
        }
    
        std::array<FresnelZone, VCT::Constants::MaximumNumberOfInteractions> zones;
        int32_t numZones;
    };

    bool IsInExistingPathZone(const TraceData& traceData, const Transmitter& transmitter, const Receiver& receiver, float waveLength, std::vector<VCT::PathFresnelZones>& zones)
    {
        bool result = false;
        for (auto& zone : zones)
        {
            if (zone.IsSharedZone(traceData, transmitter.position, receiver.position))
            {
                result = true;
                break;
            }
        }
        zones.emplace_back(traceData, transmitter, receiver, waveLength);
        return result;
    }

    void PathStorage::TryRemoveDuplicates(uint32_t txID, uint32_t rxID, const Transmitter& transmitter, const Receiver& receiver, float waveLength)
    {
        size_t txRxHash = CalculateHash(txID, rxID);
        auto it = m_PathMap.find(txRxHash);
        if (it != m_PathMap.end())
        {
            std::vector<TraceData>& paths = it->second;
            std::vector<PathFresnelZones> pathFresnelZones;
            pathFresnelZones.reserve(paths.size());

            std::vector<TraceData> newPaths;
            newPaths.reserve(paths.size());

            std::sort(paths.begin(), paths.end(), [](const TraceData& a, const TraceData& b) { return a.timeDelay < b.timeDelay; });

            for (const TraceData& path : paths)
            {
                if (path.numInteractions == 0 || !IsInExistingPathZone(path, transmitter, receiver, waveLength, pathFresnelZones))
                    newPaths.push_back(path);
            }
            it->second = std::move(newPaths);
        }
    }

    const std::vector<TraceData>* PathStorage::GetPaths(uint32_t txID, uint32_t rxID) const
    {
        auto it = m_PathMap.find(CalculateHash(txID, rxID));
        return it != m_PathMap.end() ? &it->second : nullptr;
    }
}