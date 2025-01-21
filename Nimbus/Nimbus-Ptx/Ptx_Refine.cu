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

extern "C" __global__ void __intersection__ST()
{
	OnIntersect(data.rtParams);
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
			result.pathType = Nimbus::PathType::Specular;
			result.rxID = rxID;
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

extern "C" __global__ void __raygen__RefineDiffraction()
{

}