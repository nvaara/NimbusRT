#pragma once
#include "Nimbus/Types.hpp"
#include <array>

namespace Nimbus
{
	struct PathHashMapInfo
	{
		PathHashMapInfo(double timeDelay, uint32_t pathIndex);
		double timeDelay;
		uint32_t pathIndex;
	};

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
		void Reserve(uint32_t maxNumInteractions, size_t numReceivers, size_t numTransmitters, uint32_t maxLinkPaths, SionnaPathType pathType);

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
		void ReservePaths();

		std::array<uint32_t, SionnaPathData::PathTypeCount> maxNumIa;
		std::vector<glm::vec3> transmitters;
		std::vector<glm::vec3> receivers;
		std::array<uint32_t, PathTypeCount> maxLinkPaths;
		std::array<SionnaPathTypeData, PathTypeCount> paths;
	};
}