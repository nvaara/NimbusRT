#include "PathData.hpp"

namespace Nimbus
{
	PathHashMapInfo::PathHashMapInfo(double timeDelay, uint32_t pathIndex)
		: timeDelay(timeDelay)
		, pathIndex(pathIndex)
	{
	}

    void SionnaPathTypeData::Reserve(uint32_t maxNumInteractions, size_t numReceivers, size_t numTransmitters, uint32_t maxLinkPaths, SionnaPathType pathType)
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

	void SionnaPathData::ReservePaths()
	{
		for (uint32_t i = 1; i < PathTypeCount; ++i) //Skip LOS because its included in Specular paths
			paths[i].Reserve(maxNumIa[i], receivers.size(), transmitters.size(), maxLinkPaths[i], static_cast<SionnaPathType>(i));
	}
}