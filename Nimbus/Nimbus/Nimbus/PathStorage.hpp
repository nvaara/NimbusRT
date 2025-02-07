#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include "Nimbus/Types.hpp"
#include "Hash.hpp"
#include <array>
#include "Environment.hpp"

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
			void Reserve(uint32_t maxNumInteractions, size_t numReceivers, size_t numTransmitters, uint32_t maxLinkPaths, SionnaPathType pathType)
			{
				size_t interactionBufferElements = maxNumInteractions * numReceivers * numTransmitters * maxLinkPaths;
				size_t interactionBufferElementsIncident = (maxNumInteractions + 1) * numReceivers * numTransmitters * maxLinkPaths;
				size_t pathBufferElements = numReceivers * numTransmitters * maxLinkPaths;

				interactions.resize(interactionBufferElements);
				normals.resize(interactionBufferElements * (pathType == SionnaPathType::Diffracted ? 2u : 1u));
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

				switch (pathType)
				{
				case SionnaPathType::Scattered:
				{
					scattering.lastObjects.resize(pathBufferElements);
					scattering.lastVertices.resize(pathBufferElements);
					scattering.lastNormal.resize(pathBufferElements);
					scattering.lastIncident.resize(pathBufferElements);
					scattering.lastDeflected.resize(pathBufferElements);
					scattering.distToLastIa.resize(pathBufferElements);
					scattering.distFromLastIaToRx.resize(pathBufferElements);
					break;
				}
				case SionnaPathType::RIS:
				{
					ris.cosThetaTx.resize(pathBufferElements);
					ris.cosThetaRx.resize(pathBufferElements);
					ris.distanceTxRis.resize(pathBufferElements);
					ris.distanceRxRis.resize(pathBufferElements);
					break;
				}
				default:
					break;
				}
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

			struct 
			{
				std::vector<uint32_t> lastObjects;
				std::vector<glm::vec3> lastVertices;
				std::vector<glm::vec3> lastNormal;
				std::vector<glm::vec3> lastIncident;
				std::vector<glm::vec3> lastDeflected;
				std::vector<float> distToLastIa;
				std::vector<float> distFromLastIaToRx;

			} scattering;
			struct
			{
				std::vector<float> cosThetaTx;
				std::vector<float> cosThetaRx;
				std::vector<float> distanceTxRis;
				std::vector<float> distanceRxRis;
			} ris;
		};

		struct SionnaPathData
		{
			static constexpr size_t PathTypeCount = static_cast<size_t>(SionnaPathType::TypeCount);
			void ReservePaths()
			{
				for (uint32_t i = 1; i < PathTypeCount; ++i) //Skip LOS because its included in Specular paths
					paths[i].Reserve(maxNumIa[i], receivers.size(), transmitters.size(), maxLinkPaths[i], static_cast<SionnaPathType>(i));
			}
			std::array<uint32_t, SionnaPathData::PathTypeCount> maxNumIa;
			std::vector<glm::vec3> transmitters;
			std::vector<glm::vec3> receivers;
			std::array<uint32_t, PathTypeCount> maxLinkPaths;
			std::array<SionnaPathTypeData, PathTypeCount> paths;
		};

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
		std::unordered_map<PathHashKey, PathHashMapInfo> m_PathMap;
		std::vector<InteractionData> m_InteractionData;

		std::vector<double> m_TimeDelays;
		std::vector<uint32_t> m_TxIDs;
		std::vector<uint32_t> m_RxIDs;
		std::vector<PathType> m_PathTypes;
		std::vector<uint8_t> m_NumInteractions;
	};
}