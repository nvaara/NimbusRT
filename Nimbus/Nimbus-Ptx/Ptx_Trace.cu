#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "HitGroupCommon.cuh"
#include "Ray.cuh"

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

inline __device__ void SaveLosPath(uint32_t txID, uint32_t rxID)
{
	Nimbus::PathInfo pathInfo{};
	pathInfo.txID = txID;
	pathInfo.rxID = rxID;
	pathInfo.timeDelay = glm::length(data.receivers[rxID] - data.transmitters[txID]) * Nimbus::Constants::InvLightSpeedInVacuum;
	pathInfo.pathType = Nimbus::PathType::LineOfSight;
	pathInfo.numInteractions = 0;
	data.receivedPathData.pathInfos[atomicAdd(data.receivedPathCount, 1u)] = pathInfo;
}

extern "C" __global__ void __raygen__TransmitLOS()
{
	uint32_t rxID = optixGetLaunchIndex().x;
	Ray ray = Ray(data.transmitters[data.currentTxID], data.receivers[rxID]);
	if (!ray.Trace(data.rtParams.env.asHandle, 0.0f, 0.0f))
		SaveLosPath(data.currentTxID, rxID);
}

inline __device__ void WriteInteractionData(const Ray& ray, Nimbus::PathInfoST& pathInfo, uint32_t pathIndex)
{
	uint32_t index = pathIndex * data.maxNumIa + pathInfo.numInteractions++;
	data.propagationPathData.interactions[index] = ray.GetOrigin() + ray.GetDirection() * ray.GetPayload().t;
	data.propagationPathData.normals[index] = ray.GetPayload().normal;
	data.propagationPathData.labels[index] = ray.GetPayload().label;
	data.propagationPathData.materials[index] = ray.GetPayload().material;

	pathInfo.rtPointIndex = ray.GetPayload().rtPointIndex;
	pathInfo.direction = glm::reflect(glm::normalize(data.propagationPathData.interactions[index] - ray.GetOrigin()), data.propagationPathData.normals[index]);
	pathInfo.timeDelay += ray.GetPayload().t * Nimbus::Constants::InvLightSpeedInVacuum;
	pathInfo.terminated = false;
}

inline __device__ void InitPath(const Ray& ray, Nimbus::PathType pathType)
{
	Nimbus::PathInfoST pathInfo{};
	pathInfo.txID = data.currentTxID;
	pathInfo.pathType = pathType;
	uint32_t pathIndex = atomicAdd(data.pathCount, 1u);
	WriteInteractionData(ray, pathInfo, pathIndex);
	data.propagationPathData.pathInfos[pathIndex] = pathInfo;
}

extern "C" __global__ void __raygen__Transmit()
{
	uint32_t rtPointIndex = optixGetLaunchIndex().x;
	Ray ray = Ray(data.transmitters[data.currentTxID], data.rtParams.env.rtPoints[rtPointIndex]);
	if (ray.Trace(data.rtParams.env.asHandle, data.rtParams.rayBias, data.rtParams.rayBias) && ray.GetPayload().rtPointIndex == rtPointIndex)
		InitPath(ray, Nimbus::PathType::Specular);
}

extern "C" __global__ void __raygen__Propagate()
{
	uint32_t pathIndex = optixGetLaunchIndex().x;
	Nimbus::PathInfoST& path = data.propagationPathData.pathInfos[pathIndex];

	if (path.terminated)
		return;

	const glm::vec3& pos = data.propagationPathData.interactions[data.maxNumIa * optixGetLaunchIndex().x + path.numInteractions - 1];
	Ray ray = Ray(pos, pos + path.direction * data.rtParams.rayMaxLength);

	if (ray.Trace(data.rtParams.env.asHandle, data.rtParams.rayBias, data.rtParams.rayBias) && !ray.IsHitRIS())
		WriteInteractionData(ray, path, pathIndex);
	else
		path.terminated = true;
}