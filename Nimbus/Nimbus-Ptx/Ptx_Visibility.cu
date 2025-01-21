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

extern "C" __global__ void __intersection__ST()
{
	OnIntersect(data.rtParams);
}

extern "C" __global__ void __closesthit__ST_TR()
{
	OnClosestHitTriangle(data.rtParams.env);
}

extern "C" __global__ void __raygen__Visibility()
{
	uint32_t rtPointIndex = optixGetLaunchIndex().x;
	uint32_t rxID = optixGetLaunchIndex().y;
	
	Ray ray = Ray(data.receivers[rxID], data.rtParams.env.rtPoints[rtPointIndex]);
	if (ray.Trace(data.rtParams.env.asHandle, data.rtParams.rayBias, data.rtParams.env.vwInfo.size) && ray.GetPayload().rtPointIndex == rtPointIndex)
	{
		glm::uvec2 scatterMask = Nimbus::Utils::GetBinBitMask32(rtPointIndex);
		glm::uvec2 rxMask = Nimbus::Utils::GetBinBitMask32(rtPointIndex * data.numRx + rxID);
		atomicOr(&data.scattererVisible[scatterMask.x], scatterMask.y);
		atomicOr(&data.rxVisible[rxMask.x], rxMask.y);
	}
}