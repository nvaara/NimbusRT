#pragma once
#include "Types.hpp"
#include <unordered_set>
#include <unordered_map>

namespace VCT
{
    class PathStorage
    {
    public:
        PathStorage(uint32_t pathsPerHash = 10);

        void AddPaths(const std::vector<TraceData>& traceDatas, bool useHash);
        void AddPaths(const TraceData* traceDatas, uint32_t numPaths, bool useHash);
        void AddPath(const TraceData& traceData, bool useHash);
        void TryRemoveDuplicates(uint32_t txID, uint32_t rxID, const Transmitter& transmitter, const Receiver& receiver, float waveLength);

        const std::vector<TraceData>* GetPaths(uint32_t txID, uint32_t rxID) const;

    private:
        struct PathReference
        {
            uint32_t longestPathIndex = 0;
            float maxTimeDelay = 0.0f;
            std::vector<uint32_t> references;
        };

    private:
        uint32_t m_PathsPerHash;
        std::unordered_map<size_t, PathReference> m_PathHashReferenceMap;
        std::unordered_map<size_t, std::vector<TraceData>> m_PathMap;
    };
}