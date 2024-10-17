#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include "Nimbus/Types.hpp"
#include "Hash.hpp"
#include <array>

namespace Nimbus
{
	struct PathHashKey
	{
		PathHashKey(size_t key) : key(key) {}
		bool operator==(const PathHashKey& k) const { return k.key == key; }
		size_t key;
	};

}

template <> struct std::hash<Nimbus::PathHashKey>
{
	inline size_t operator()(const Nimbus::PathHashKey& k) const
	{
		return k.key;
	}
};

namespace Nimbus
{
	class PathStorage
	{
	public:
		struct PathHashMapInfo
		{
			PathHashMapInfo(double timeDelay, uint32_t pathIndex) : timeDelay(timeDelay), pathIndex(pathIndex) {}
			double timeDelay;
			uint32_t pathIndex;
		};

		PathStorage(uint32_t maxNumInteractions, const glm::vec3* txs, uint32_t txCount, const glm::vec3* rxs, uint32_t rxCount);
		PathHashKey GetPathHash(const PathInfo& pathInfo, const uint32_t* labels) const;

		void AddPaths(uint32_t numPaths,
					  const std::vector<PathInfo>& pathInfos,
					  const std::vector<glm::vec3>& interactions,
					  const std::vector<glm::vec3>& normals,
					  const std::vector<uint32_t>& labels,
					  const std::vector<uint32_t>& materials);

		struct PathData
		{
			uint32_t maxNumIa;
			uint32_t maxLinkPaths;
			std::vector<glm::vec3> transmitters;
			std::vector<glm::vec3> receivers;
			std::vector<glm::vec3> interactions;
			std::vector<glm::vec3> normals;
			std::vector<uint32_t> labels;
			std::vector<uint32_t> materials;
			std::vector<double> timeDelays;
			std::vector<uint32_t> txIDs;
			std::vector<uint32_t> rxIDs;
			std::vector<PathType> pathTypes;
			std::vector<uint8_t> numInteractions;
		};

		enum class SionnaPathType
		{
			Los = 0,
			Specular = 1,
			Diffracted = 2,
			Scattered = 3,
			RIS = 4,
			TypeCount = 5
		};

		struct SionnaPathTypeData
		{
			void Reserve(uint32_t maxNumInteractions, size_t numReceivers, size_t numTransmitters, uint32_t maxLinkPaths)
			{
				size_t interactionBufferElements = maxNumInteractions * numReceivers * numTransmitters * maxLinkPaths;
				size_t interactionBufferElementsIncident = (maxNumInteractions + 1) * numReceivers * numTransmitters * maxLinkPaths;
				size_t pathBufferElements = numReceivers * numTransmitters * maxLinkPaths;

				interactions.resize(interactionBufferElements);
				normals.resize(interactionBufferElements);
				materials.resize(interactionBufferElements);
				incidentRays.resize(interactionBufferElementsIncident);
				deflectedRays.resize(interactionBufferElements);

				timeDelays.resize(pathBufferElements, -1.0f);
				totalDistance.resize(pathBufferElements, -1.0f);
				mask.resize(pathBufferElements, 0u);
				kTx.resize(pathBufferElements);
				kRx.resize(pathBufferElements);

				aodElevation.resize(pathBufferElements);
				aodAzimuth.resize(pathBufferElements);
				aoaElevation.resize(pathBufferElements);
				aoaAzimuth.resize(pathBufferElements);
			}

			std::vector<glm::vec3> interactions;
			std::vector<glm::vec3> normals;
			std::vector<uint32_t> materials;
			std::vector<glm::vec3> incidentRays;
			std::vector<glm::vec3> deflectedRays;
			std::vector<float> timeDelays;
			std::vector<float> totalDistance;
			std::vector<uint8_t> mask;
			std::vector<glm::vec3> kTx;
			std::vector<glm::vec3> kRx;
			std::vector<float> aodElevation;
			std::vector<float> aodAzimuth;
			std::vector<float> aoaElevation;
			std::vector<float> aoaAzimuth;
		};

		struct SionnaPathData
		{
			static constexpr size_t PathTypeCount = static_cast<size_t>(SionnaPathType::TypeCount);
			void ReservePaths()
			{
				for (uint32_t i = 0; i < PathTypeCount; ++i)
					paths[i].Reserve(maxNumIa, receivers.size(), transmitters.size(), maxLinkPaths[i]);
			}
			uint32_t maxNumIa;
			std::vector<glm::vec3> transmitters;
			std::vector<glm::vec3> receivers;
			std::array<uint32_t, PathTypeCount> maxLinkPaths;
			std::array<SionnaPathTypeData, PathTypeCount> paths;
		};

		PathData ToPathData();
		SionnaPathData ToSionnaPathData();
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
		uint32_t m_MaxNumInteractions;
		std::vector<glm::vec3> m_Transmitters;
		std::vector<glm::vec3> m_Receivers;
		std::array<uint32_t, SionnaPathData::PathTypeCount> m_MaxLinkPaths;
		std::vector<std::array<uint32_t, SionnaPathData::PathTypeCount>> m_PathCounts;
		std::unordered_map<PathHashKey, PathHashMapInfo> m_PathMap;
		std::vector<InteractionData> m_InteractionData;

		std::vector<double> m_TimeDelays;
		std::vector<uint32_t> m_TxIDs;
		std::vector<uint32_t> m_RxIDs;
		std::vector<PathType> m_PathTypes;
		std::vector<uint8_t> m_NumInteractions;
	};
}