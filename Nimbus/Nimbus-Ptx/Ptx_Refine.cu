#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "HitGroupCommon.cuh"
#include "Ray.cuh"
#include "PathRefiner.cuh"

__constant__ Nimbus::STRTData data;

extern "C" __global__ void __miss__ST()
{

}

extern "C" __global__ void __closesthit__ST()
{
	OnClosestHit(data.rtParams.env);
}

extern "C" __global__ void __closesthit__ST_TR()
{
	OnClosestHitTriangle(data.rtParams.env);
}

extern "C" __global__ void __closesthit__ST_RIS()
{
	OnClosestHitRIS(data.rtParams.env);
}

extern "C" __global__ void __intersection__ST()
{
	OnIntersectExplicit(data.rtParams);
}

inline __device__ bool AddReceivedScatteredPath(const Nimbus::PathInfo& pathInfo, uint32_t pathDataIndex)
{
	uint32_t index = atomicAdd(data.receivedPathCount, 1u);
	bool result = index < data.numReceivedPathsMax;

	if (result)
	{
		data.receivedPathData.pathInfos[index] = pathInfo;
		uint32_t recvDataIndex = index * data.maxNumIa;
		for (uint32_t i = 0; i < pathInfo.numInteractions; ++i)
		{
			data.receivedPathData.interactions[recvDataIndex + i] = data.propagationPathData.interactions[pathDataIndex + i];
			data.receivedPathData.normals[recvDataIndex + i] = data.propagationPathData.normals[pathDataIndex + i];
			data.receivedPathData.labels[recvDataIndex + i] = data.propagationPathData.labels[pathDataIndex + i];
			data.receivedPathData.materials[recvDataIndex + i] = data.propagationPathData.materials[pathDataIndex + i];
		}
		uint32_t scatterLabel = Nimbus::Utils::WorldToVoxelID(data.receivedPathData.interactions[recvDataIndex + pathInfo.numInteractions - 1], data.rtParams.env.vwInfo);
		data.receivedPathData.labels[recvDataIndex + pathInfo.numInteractions - 1] = scatterLabel;
	}
	return result;
}

inline __device__ bool AddReceivedRefinedPath(const Nimbus::PathInfo& pathInfo, const RefineData* refineData)
{
	uint32_t index = atomicAdd(data.receivedPathCount, 1u);
	bool result = index < data.numReceivedPathsMax;
	if (result)
	{
		data.receivedPathData.pathInfos[index] = pathInfo;
		uint32_t recvDataIndex = index * data.maxNumIa;
		for (uint32_t i = 0; i < pathInfo.numInteractions; ++i)
		{
			data.receivedPathData.interactions[recvDataIndex + i] = refineData[i + 1].position;
			data.receivedPathData.normals[recvDataIndex + i] = refineData[i + 1].normal;
			data.receivedPathData.labels[recvDataIndex + i] = refineData[i + 1].label;
			data.receivedPathData.materials[recvDataIndex + i] = refineData[i + 1].material;
		}
	}
	return result;
}

inline __device__ bool AddReceivedRISPath(const Nimbus::PathInfo& pathInfo, const glm::vec3& hitPoint, const glm::vec3& normal, uint32_t label, uint32_t objectId)
{
	uint32_t index = atomicAdd(data.receivedPathCount, 1u);
	bool result = index < data.numReceivedPathsMax;
	if (result)
	{
		data.receivedPathData.pathInfos[index] = pathInfo;
		uint32_t recvDataIndex = index * data.maxNumIa;
		data.receivedPathData.interactions[recvDataIndex] = hitPoint;
		data.receivedPathData.normals[recvDataIndex] = normal;
		data.receivedPathData.labels[recvDataIndex] = label;
		data.receivedPathData.materials[recvDataIndex] = objectId;
	}
	return result;
}

extern "C" __global__ void __raygen__RefineSpecular()
{
	uint32_t pathID = optixGetLaunchIndex().x;
	uint32_t rxID = optixGetLaunchIndex().y;
	
	Nimbus::PathInfoST path = data.propagationPathData.pathInfos[pathID];

	if (path.terminated)
		return;

	glm::uvec2 pMask = Nimbus::Utils::GetBinBitMask32(pathID * data.numRx + rxID);
	glm::uvec2 rxMask = Nimbus::Utils::GetBinBitMask32(path.rtPointIndex * data.numRx + rxID);

	if (data.pathsProcessed[pMask.x] & pMask.y)
		return;

	if (data.rxVisible[rxMask.x] & rxMask.y)
	{
		Nimbus::PathInfo result{};
		uint32_t dataIndex = pathID * data.maxNumIa;
		const glm::vec3* interactions = &data.propagationPathData.interactions[dataIndex];
		const glm::vec3* normals = &data.propagationPathData.normals[dataIndex];

		PathRefiner refiner = PathRefiner(path, interactions, normals, data.transmitters[path.txID], data.receivers[rxID], rxID, data.rtParams);
		if (refiner.Refine(data.refineParams, result))
		{
			if (!AddReceivedRefinedPath(result, refiner.GetRefineData()))
				return;
		}
		atomicOr(&data.pathsProcessed[pMask.x], pMask.y);
	}
}

inline __device__ bool ValidateVisibility(uint32_t srcIndex, uint32_t dstIndex)
{
	glm::vec3 src = data.propagationPathData.interactions[srcIndex];
	glm::vec3 dst = data.propagationPathData.interactions[dstIndex];
	glm::vec3 dstNormal = data.propagationPathData.normals[dstIndex];
	uint32_t  dstLabel = data.propagationPathData.labels[dstIndex];
	Ray ray(src, dst);
	bool res = ray.Trace(data.rtParams.env.asHandle, data.rtParams.rayBias, data.rtParams.rayBias);
	bool validSpecRefl = glm::dot(ray.GetPayload().normal, dstNormal) > 0.99f;
	bool equalLabel = dstLabel == ray.GetPayload().label;
	return res && validSpecRefl && equalLabel;
}

extern "C" __global__ void __raygen__RefineScatterer()
{
	uint32_t pathID = optixGetLaunchIndex().x;
	uint32_t rxID = optixGetLaunchIndex().y;

	Nimbus::PathInfoST path = data.propagationPathData.pathInfos[pathID];

	if (path.terminated)
		return;

	glm::uvec2 pMask = Nimbus::Utils::GetBinBitMask32(pathID * data.numRx + rxID);

	uint32_t pathDataIndex = pathID * data.maxNumIa;

	uint32_t lastIaIndex = pathDataIndex + path.numInteractions - 1u;
	glm::vec3 lastIaPos = data.propagationPathData.interactions[lastIaIndex];
	glm::vec3 lastIaNormal = data.propagationPathData.normals[lastIaIndex];

	glm::uvec2 rxMask = Nimbus::Utils::GetBinBitMask32(path.rtPointIndex * data.numRx + rxID);

	if (data.pathsProcessed[pMask.x] & pMask.y)
		return;

	if (data.rxVisible[rxMask.x] & rxMask.y)
	{
		glm::vec3 rx = data.receivers[rxID];
		path.timeDelay += glm::length(rx - lastIaPos) * Nimbus::Constants::InvLightSpeedInVacuum;

		path.pathType = Nimbus::PathType::Scattering;
		path.rxID = rxID;
		uint32_t label = data.propagationPathData.labels[lastIaIndex];

		Ray ray = Ray(rx, lastIaPos);
		bool res = ray.Trace(data.rtParams.env.asHandle, 0.0f, data.rtParams.rayBias);

		if (!res || glm::dot(ray.GetPayload().normal, lastIaNormal) < 0.99f || ray.GetPayload().label != label)
		{
			atomicOr(&data.pathsProcessed[pMask.x], pMask.y);
			return;
		}

		for (uint32_t i = 0; i < path.numInteractions - 1; ++i)
		{
			if (!ValidateVisibility(pathDataIndex + i + 1, pathDataIndex + i))
			{
				atomicOr(&data.pathsProcessed[pMask.x], pMask.y);
				return;
			}
		}
		if (AddReceivedScatteredPath(path, pathDataIndex))
			atomicOr(&data.pathsProcessed[pMask.x], pMask.y);
	}
}

inline __device__ bool ValidateDiffraction(const Nimbus::DiffractionEdge& edge, const Nimbus::PathInfo& pathInfo, const RefineData* refineData)
{
	glm::vec3 edgePoint = refineData[1].position;
	Ray txToEdge = Ray(refineData[0].position, edgePoint);
	Ray rxToEdge = Ray(refineData[2].position, edgePoint);
	constexpr uint32_t flags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
	bool txEdgeVisible = !txToEdge.Trace(data.rtParams.env.asHandle, 0.0f, -data.rtParams.rayBias, flags);
	bool rxEdgeVisible = !rxToEdge.Trace(data.rtParams.env.asHandle, 0.0f, -data.rtParams.rayBias, flags);
	bool pointOnEdge = glm::length(edge.midPoint - edgePoint) <= edge.halfLength;
	bool isValidRay = edge.IsValidIncidentRayForDiffraction(txToEdge.GetDirection());
	return txEdgeVisible && rxEdgeVisible && pointOnEdge && isValidRay;
}

extern "C" __global__ void __raygen__RefineDiffraction()
{
	uint32_t edgeID = optixGetLaunchIndex().x;
	uint32_t rxID = optixGetLaunchIndex().y;
	
	const Nimbus::DiffractionEdge& edge = data.rtParams.env.edges[edgeID];
	glm::vec3 iaPoint = (edge.start + edge.end) * 0.5f;
	glm::vec3 normal = glm::normalize(edge.end - edge.start);

	glm::uvec2 edgeMask = Nimbus::Utils::GetBinBitMask32(edgeID * data.numRx + rxID);

	if (data.pathsProcessed[edgeMask.x] & edgeMask.y)
		return;

	Nimbus::PathInfoST path{};
	path.txID = data.currentTxID;
	path.pathType = Nimbus::PathType::Diffraction;
	path.numInteractions = 1u;

	Nimbus::PathInfo result{};
	PathRefiner refiner = PathRefiner(path, &iaPoint, &normal, data.transmitters[path.txID], data.receivers[rxID], rxID, data.rtParams);
	if (refiner.Refine(data.refineParams, result))
	{
		if (ValidateDiffraction(edge, result, refiner.GetRefineData()) && !AddReceivedRefinedPath(result, refiner.GetRefineData()))
			return;
	}
	atomicOr(&data.pathsProcessed[edgeMask.x], edgeMask.y);
}

extern "C" __global__ void __raygen__ComputeRisPaths()
{
	uint32_t cellIndex = optixGetLaunchIndex().x;
	uint32_t rxID = optixGetLaunchIndex().y;
	glm::vec3 tx = data.transmitters[data.currentTxID];
	glm::vec3 rx = data.receivers[rxID];
	glm::vec3 cell = data.rtParams.env.ris.cellWorldPositions[cellIndex];

	glm::uvec2 cellMask = Nimbus::Utils::GetBinBitMask32(cellIndex * data.numRx + rxID);
	if (data.pathsProcessed[cellMask.x] & cellMask.y)
		return;

	Ray rayTx = Ray(tx, cell);
	Ray rayRx = Ray(rx, cell);
	bool txHit = rayTx.Trace(data.rtParams.env.asHandle, data.rtParams.rayBias, data.rtParams.rayBias);
	bool rxHit = rayRx.Trace(data.rtParams.env.asHandle, data.rtParams.rayBias, data.rtParams.rayBias);

	if (txHit && rxHit && rayTx.IsHitRIS() && rayRx.IsHitRIS())
	{
		const Ray::Payload& payloadTx = rayTx.GetPayload();
		const Ray::Payload& payloadRx = rayRx.GetPayload();

		glm::vec3 hitPointTx = rayTx.GetOrigin() + rayTx.GetDirection() * payloadTx.t;
		glm::vec3 hitPointRx = rayRx.GetOrigin() + rayRx.GetDirection() * payloadRx.t;
		glm::vec3 diff = hitPointRx - hitPointTx;
		float distDiff = glm::abs(glm::dot(diff, diff));
		constexpr float distDiffThreshold = 1e-4f;
		
		bool validSideTx = glm::dot(rayTx.GetDirection(), payloadTx.normal) < 0.0f;
		bool validSideRx = glm::dot(rayRx.GetDirection(), payloadRx.normal) < 0.0f;
		bool sameLabel = payloadTx.label == payloadRx.label;
		if (validSideTx && validSideRx && sameLabel && distDiff < distDiffThreshold)
		{
			Nimbus::PathInfo path{};
			path.txID = data.currentTxID;
			path.pathType = Nimbus::PathType::RIS;
			path.numInteractions = 1u;
			path.rxID = rxID;
			path.timeDelay = (payloadTx.t + payloadRx.t) * Nimbus::Constants::InvLightSpeedInVacuum;

			uint32_t label = cellIndex;
			uint32_t objectId = payloadTx.label;
			if (!AddReceivedRISPath(path, hitPointTx, payloadTx.normal, label, objectId))
				return;
		}
	}
	atomicOr(&data.pathsProcessed[cellMask.x], cellMask.y);
}