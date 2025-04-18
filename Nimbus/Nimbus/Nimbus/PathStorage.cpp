#include "PathStorage.hpp"
#include <iostream>

namespace Nimbus
{
	PathStorage::PathStorage(uint32_t maxNumInteractions, const glm::vec3* txs, uint32_t txCount, const glm::vec3* rxs, uint32_t rxCount)
		: m_MaxNumInteractions({ maxNumInteractions, maxNumInteractions, 1u, maxNumInteractions, 1u })
		, m_InteractionData(maxNumInteractions)
		, m_Transmitters(txs, txs + txCount)
		, m_Receivers(rxs, rxs + rxCount)
		, m_PathCounts(txCount * rxCount)
		, m_MaxLinkPaths()
	{
		for (auto& counts : m_PathCounts)
			counts = {};

		for (auto& counts : m_MaxLinkPaths)
			counts = {};
	}

	void PathStorage::AddPaths(uint32_t numPaths,
							   const std::vector<PathInfo>& pathInfos,
							   const std::vector<glm::vec3>& interactions,
							   const std::vector<glm::vec3>& normals,
							   const std::vector<uint32_t>& labels,
							   const std::vector<uint32_t>& materials)
	{
		for (size_t pathIndex = 0; pathIndex < numPaths; ++pathIndex)
		{
			const PathInfo& pathInfo = pathInfos[pathIndex];
			size_t dataIndex = pathIndex * m_InteractionData.size();
			PathHash hash = PathHash(pathInfo, &labels[dataIndex]);

			auto [it, emplaced] = m_PathMap.try_emplace(hash, pathInfo.timeDelay, static_cast<uint32_t>(m_PathMap.size()));
			if (emplaced)
			{
				for (uint32_t iaIndex = 0; iaIndex < m_InteractionData.size(); ++iaIndex)
				{
					if (iaIndex < pathInfo.numInteractions)
					{
						m_InteractionData[iaIndex].interactions.emplace_back(interactions[dataIndex + iaIndex]);
						m_InteractionData[iaIndex].normals.emplace_back(normals[dataIndex + iaIndex]);
						m_InteractionData[iaIndex].labels.emplace_back(labels[dataIndex + iaIndex]);
						m_InteractionData[iaIndex].materials.emplace_back(materials[dataIndex + iaIndex]);
					}
					else
					{
						m_InteractionData[iaIndex].interactions.emplace_back(0.0f);
						m_InteractionData[iaIndex].normals.emplace_back(0.0f);
						m_InteractionData[iaIndex].labels.emplace_back(~0u);
						m_InteractionData[iaIndex].materials.emplace_back(~0u);
					}
				}
				uint32_t typeIndex = static_cast<uint32_t>(GetSionnaPathType(pathInfo.pathType));
				m_MaxLinkPaths[typeIndex] = glm::max(m_MaxLinkPaths[typeIndex], ++m_PathCounts[pathInfo.rxID * m_Transmitters.size() + pathInfo.txID][typeIndex]);
				m_TimeDelays.emplace_back(pathInfo.timeDelay);
				m_TxIDs.emplace_back(pathInfo.txID);
				m_RxIDs.emplace_back(pathInfo.rxID);
				m_PathTypes.emplace_back(pathInfo.pathType);
				m_NumInteractions.emplace_back(pathInfo.numInteractions);
			}
			else if (it->second.timeDelay > pathInfo.timeDelay)
			{
				it->second.timeDelay = pathInfo.timeDelay;
				for (uint32_t iaIndex = 0; iaIndex < pathInfo.numInteractions; ++iaIndex)
				{
					m_InteractionData[iaIndex].interactions[it->second.pathIndex] = interactions[dataIndex + iaIndex];
					m_InteractionData[iaIndex].normals[it->second.pathIndex] = normals[dataIndex + iaIndex];
					m_InteractionData[iaIndex].labels[it->second.pathIndex] = labels[dataIndex + iaIndex];
					m_InteractionData[iaIndex].materials[it->second.pathIndex] = materials[dataIndex + iaIndex];
				}
				m_TimeDelays[it->second.pathIndex] = pathInfo.timeDelay;
				m_NumInteractions[it->second.pathIndex] = pathInfo.numInteractions;
			}
		}
	}

	PathData PathStorage::ToPathData()
	{
		PathData data{};
		data.maxNumIa = m_MaxNumInteractions[static_cast<uint32_t>(SionnaPathType::Specular)];
		data.maxLinkPaths = 0u;
		data.transmitters = m_Transmitters;
		data.receivers = m_Receivers;

		for (uint32_t i = 0; i < m_InteractionData.size(); ++i)
		{
			data.interactions.reserve(m_InteractionData.size() * m_InteractionData[0].interactions.size());
			data.normals.reserve(m_InteractionData.size() * m_InteractionData[0].interactions.size());
			data.labels.reserve(m_InteractionData.size() * m_InteractionData[0].interactions.size());
			data.materials.reserve(m_InteractionData.size() * m_InteractionData[0].interactions.size());
		}

		for (uint32_t i = 0; i < m_InteractionData.size(); ++i)
		{
			data.interactions.insert(data.interactions.end(), m_InteractionData[i].interactions.begin(), m_InteractionData[i].interactions.end());
			data.normals.insert(data.normals.end(), m_InteractionData[i].normals.begin(), m_InteractionData[i].normals.end());
			data.labels.insert(data.labels.end(), m_InteractionData[i].labels.begin(), m_InteractionData[i].labels.end());
			data.materials.insert(data.materials.end(), m_InteractionData[i].materials.begin(), m_InteractionData[i].materials.end());
		}
		data.timeDelays = m_TimeDelays;
		data.txIDs = m_TxIDs;
		data.rxIDs = m_RxIDs;
		data.pathTypes = m_PathTypes;
		data.numInteractions = m_NumInteractions;
		return data;
	}

	SionnaPathData PathStorage::ToSionnaPathData(const Environment& env)
	{
		SionnaPathData sionnaData{};
		float voxelSize = env.GetVoxelSize();
		float voxelArea = voxelSize * voxelSize;

		sionnaData.transmitters = m_Transmitters;
		sionnaData.receivers = m_Receivers;
		sionnaData.maxNumIa = m_MaxNumInteractions;
		sionnaData.maxLinkPaths = m_MaxLinkPaths;
		if (m_MaxLinkPaths[static_cast<uint32_t>(SionnaPathType::RIS)] > 0u)
			sionnaData.maxLinkPaths[static_cast<uint32_t>(SionnaPathType::RIS)] = env.GetRisPointCount();
		sionnaData.ReservePaths();

		uint32_t numTotalPaths = static_cast<uint32_t>(m_TxIDs.size());
		for (uint32_t pathIndex = 0; pathIndex < numTotalPaths; ++pathIndex)
		{
			SionnaPathType pathType = GetSionnaPathType(m_PathTypes[pathIndex]);
			uint32_t typeIndex = static_cast<uint32_t>(pathType);
			uint32_t txID = m_TxIDs[pathIndex];
			uint32_t rxID = m_RxIDs[pathIndex];
			glm::vec3 tx = m_Transmitters[txID];
			glm::vec3 rx = m_Receivers[rxID];
			uint32_t numInteractions = m_NumInteractions[pathIndex];
			double timeDelay = m_TimeDelays[pathIndex];

			size_t pathOffset = --m_PathCounts[rxID * m_Transmitters.size() + txID][typeIndex];

			if (pathType == SionnaPathType::RIS)
				pathOffset = m_InteractionData[0].labels[pathIndex];

			size_t txOffset = txID * sionnaData.maxLinkPaths[typeIndex];
			size_t rxOffset = rxID * m_Transmitters.size() * sionnaData.maxLinkPaths[typeIndex];

			for (uint32_t ia = 0; ia < m_MaxNumInteractions[typeIndex]; ++ia)
			{
				size_t iaOffset = ia * m_Receivers.size() * m_Transmitters.size() * sionnaData.maxLinkPaths[typeIndex];
				size_t pathIaIndex = iaOffset + rxOffset + txOffset + pathOffset;

				glm::vec3 iaPoint = m_InteractionData[ia].interactions[pathIndex];
				sionnaData.paths[typeIndex].interactions[pathIaIndex] = iaPoint;
				sionnaData.paths[typeIndex].normals[pathIaIndex] = m_InteractionData[ia].normals[pathIndex];
				sionnaData.paths[typeIndex].materials[pathIaIndex] = m_InteractionData[ia].materials[pathIndex];
				sionnaData.paths[typeIndex].incidentRays[pathIaIndex] = ia > 0 ? glm::normalize(iaPoint - m_InteractionData[ia - 1].interactions[pathIndex]) : glm::normalize(iaPoint - tx);
				sionnaData.paths[typeIndex].deflectedRays[pathIaIndex] = static_cast<int32_t>(ia) < static_cast<int32_t>(numInteractions) - 1 ? glm::normalize(m_InteractionData[ia + 1].interactions[pathIndex] - iaPoint) : glm::normalize(rx - iaPoint);
			}
			size_t pathDataIndex = txOffset + rxOffset + pathOffset;
			sionnaData.paths[typeIndex].timeDelays[pathDataIndex] = static_cast<float>(m_TimeDelays[pathIndex]);
			sionnaData.paths[typeIndex].totalDistance[pathDataIndex] = static_cast<float>(m_TimeDelays[pathIndex]) * Constants::LightSpeedInVacuum;
			sionnaData.paths[typeIndex].mask[pathDataIndex] = 1u;
			sionnaData.paths[typeIndex].kTx[pathDataIndex] = numInteractions > 0 ? glm::normalize(m_InteractionData[0].interactions[pathIndex] - tx) : glm::normalize(rx - tx);
			sionnaData.paths[typeIndex].kRx[pathDataIndex] = numInteractions > 0 ? glm::normalize(m_InteractionData[numInteractions - 1].interactions[pathIndex] - rx) : glm::normalize(tx - rx);
			
			size_t incidentToRxIndex = numInteractions * m_Receivers.size() * m_Transmitters.size() * sionnaData.maxLinkPaths[typeIndex];
			sionnaData.paths[typeIndex].incidentRays[incidentToRxIndex + rxOffset + txOffset + pathOffset] = -sionnaData.paths[typeIndex].kRx[pathDataIndex];

			sionnaData.paths[typeIndex].aodElevation[pathDataIndex] = glm::acos(sionnaData.paths[typeIndex].kTx[pathDataIndex].z);
			sionnaData.paths[typeIndex].aodAzimuth[pathDataIndex] = glm::atan(sionnaData.paths[typeIndex].kTx[pathDataIndex].y, sionnaData.paths[typeIndex].kTx[pathDataIndex].x);
			sionnaData.paths[typeIndex].aoaElevation[pathDataIndex] = glm::acos(sionnaData.paths[typeIndex].kRx[pathDataIndex].z);
			sionnaData.paths[typeIndex].aoaAzimuth[pathDataIndex] = glm::atan(sionnaData.paths[typeIndex].kRx[pathDataIndex].y, sionnaData.paths[typeIndex].kRx[pathDataIndex].x);

			switch (pathType)
			{
			case SionnaPathType::Scattered:
			{
				sionnaData.paths[typeIndex].scattering.lastObjects[pathDataIndex] = m_InteractionData[numInteractions - 1].materials[pathIndex];
				sionnaData.paths[typeIndex].scattering.lastVertices[pathDataIndex] = m_InteractionData[numInteractions - 1].interactions[pathIndex];
				sionnaData.paths[typeIndex].scattering.lastNormal[pathDataIndex] = m_InteractionData[numInteractions - 1].normals[pathIndex];

				size_t iaOffset = (numInteractions - 1) * m_Receivers.size() * m_Transmitters.size() * sionnaData.maxLinkPaths[typeIndex];
				size_t pathIaIndex = iaOffset + rxOffset + txOffset + pathOffset;
				sionnaData.paths[typeIndex].scattering.lastIncident[pathDataIndex] = sionnaData.paths[typeIndex].incidentRays[pathIaIndex];
				sionnaData.paths[typeIndex].scattering.lastDeflected[pathDataIndex] = sionnaData.paths[typeIndex].deflectedRays[pathIaIndex];

				glm::vec3 lastIncident = sionnaData.paths[typeIndex].scattering.lastIncident[pathDataIndex];
				glm::vec3 lastNormal = sionnaData.paths[typeIndex].scattering.lastNormal[pathDataIndex];
				float scaling = glm::max(glm::sqrt(glm::abs(glm::dot(lastNormal, lastIncident)) * voxelArea), 1e-6f);
				float totalDistance = sionnaData.paths[typeIndex].totalDistance[pathDataIndex];
				sionnaData.paths[typeIndex].scattering.distFromLastIaToRx[pathDataIndex] = glm::length(m_InteractionData[numInteractions - 1].interactions[pathIndex] - rx) / scaling;
				sionnaData.paths[typeIndex].scattering.distToLastIa[pathDataIndex] = (totalDistance - sionnaData.paths[typeIndex].scattering.distFromLastIaToRx[pathDataIndex]);
				break;
			}
			case SionnaPathType::RIS:
			{
				glm::vec3 risPoint = m_InteractionData[numInteractions - 1].interactions[pathIndex];
				glm::vec3 normal = m_InteractionData[numInteractions - 1].normals[pathIndex];
				sionnaData.paths[typeIndex].ris.cosThetaTx[pathDataIndex] = glm::dot(-sionnaData.paths[typeIndex].kTx[pathDataIndex], normal);
				sionnaData.paths[typeIndex].ris.cosThetaRx[pathDataIndex] = glm::dot(-sionnaData.paths[typeIndex].kRx[pathDataIndex], normal);
				sionnaData.paths[typeIndex].ris.distanceTxRis[pathDataIndex] = glm::length(tx - risPoint);
				sionnaData.paths[typeIndex].ris.distanceRxRis[pathDataIndex] = glm::length(rx - risPoint);
				break;
			}
			default:
				break;
			}
		}
		return sionnaData;
	}

	SionnaPathType PathStorage::GetSionnaPathType(PathType type)
	{
		switch (type)
		{
		case PathType::LineOfSight:
		case PathType::Specular:
			return SionnaPathType::Specular;
		case PathType::Diffraction:
			return SionnaPathType::Diffracted;
		case PathType::Scattering:
			return SionnaPathType::Scattered;
		case PathType::RIS:
			return SionnaPathType::RIS;
		default:
			assert(false);
			return SionnaPathType::Specular;
		}
	}
}