#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include "Nimbus/Types.hpp"
#include "Hash.hpp"
#include <array>
#include "Environment.hpp"
#include "PathData.hpp"

namespace Nimbus
{
	class PathStorage
	{
	public:
		PathStorage(uint32_t maxNumInteractions, const glm::vec3* txs, uint32_t txCount, const glm::vec3* rxs, uint32_t rxCount);
		void AddPaths(uint32_t numPaths,
					  const std::vector<PathInfo>& pathInfos,
					  const std::vector<glm::vec3>& interactions,
					  const std::vector<glm::vec3>& normals,
					  const std::vector<uint32_t>& labels,
					  const std::vector<uint32_t>& materials);

		PathData ToPathData();
		SionnaPathData ToSionnaPathData(const Environment& env);
		SionnaPathType GetSionnaPathType(PathType type);

	private:
		struct InteractionData
		{
			std::vector<glm::vec3> interactions;
			std::vector<glm::vec3> normals;
			std::vector<uint32_t> labels;
			std::vector<uint32_t> materials;
		};

	private:
		std::array<uint32_t, SionnaPathData::PathTypeCount> m_MaxNumInteractions;
		std::vector<glm::vec3> m_Transmitters;
		std::vector<glm::vec3> m_Receivers;
		std::array<uint32_t, SionnaPathData::PathTypeCount> m_MaxLinkPaths;
		std::vector<std::array<uint32_t, SionnaPathData::PathTypeCount>> m_PathCounts;
		std::unordered_map<PathHash, PathHashMapInfo> m_PathMap;
		std::vector<InteractionData> m_InteractionData;

		std::vector<double> m_TimeDelays;
		std::vector<uint32_t> m_TxIDs;
		std::vector<uint32_t> m_RxIDs;
		std::vector<PathType> m_PathTypes;
		std::vector<uint8_t> m_NumInteractions;
	};
}