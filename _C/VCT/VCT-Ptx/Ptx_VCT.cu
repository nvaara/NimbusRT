#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "Propagation.cuh"
#include "PathRefiner.cuh"
#include "HitGroupCommon.cuh"
#include "Interaction.cuh"

__constant__ VCT::VCTData data;

extern "C" __global__ void __miss__VCT()
{
	OnMiss();
}


extern "C" __global__ void __closesthit__VCT()
{
	OnClosestHit(data.sceneData.rtParams.primitiveInfos[optixGetPrimitiveIndex()].ID);
}

extern "C" __global__ void __intersection__VCT()
{
	OnIntersect(data.sceneData.rtParams);
}

struct VoxelHandler
{
	inline __device__ bool operator()(const glm::vec3& rayOrigin, const VCT::VoxelInfo& voxelInfo, const VCT::IntersectionData& intersectionData, VCT::PropagationData& parent)
	{
		float ieRadius = data.coneTracingData.ieBoundingSphereRadius;
		uint32_t startIndex = parent.voxelTraceData.localIeID;
		parent.voxelTraceData.localIeID = 0;
		uint32_t parentID = parent.tpData.traceData.interactions[parent.tpData.traceData.numInteractions - 1].ieID;
		for (uint32_t localSurfaceIndex = startIndex; localSurfaceIndex < voxelInfo.ieIndexInfo.count; ++localSurfaceIndex)
		{
			uint32_t ieID = voxelInfo.ieIndexInfo.first + localSurfaceIndex;
			const VCT::IntersectableEntity& ie = data.sceneData.intersectableEntities[ieID];
			Ray ray = Ray(rayOrigin, ie.rtPoint);
			if (parentID != ieID && IsValidInteraction(ie.type, parent.tpData, data.coneTracingData) && (intersectionData.Intersect(ie.voxelSpaceRtPoint, ieRadius, 0.0f) || SkipIntersectIE(parent, ie.type)))
			{
				if (ray.Trace(data.sceneData.rtParams, ieID, ie.type))
				{
					if (!HandleValidInteraction(ray, ie, parent.tpData))
					{
						parent.voxelTraceData.localIeID = localSurfaceIndex;
						return false;
					}
				}
			}
		}
		return true;
	}
};
extern "C" __global__ void __raygen__VCT()
{
	uint32_t launchIndex = optixGetLaunchIndex().x;
	uint32_t depthLevel = *data.coneTracingData.depthLevel;
	VCT::PropagationData& propPath = data.pathData.propPaths[depthLevel][launchIndex];
	
	if (!propPath.voxelTraceData.finished)
		Propagate<VoxelHandler>(propPath, data.coneTracingData, *data.status);
}

extern "C" __global__ void __raygen__TransmitVCT()
{
	uint32_t ieID = optixGetLaunchIndex().x;
	if (!data.transmitIndexProcessed[ieID])
	{
		VCT::TraceProcessingData tpData{};
		tpData.traceData.transmitterID = data.currentTransmitterID;
		const VCT::Transmitter& tx = data.sceneData.transmitters[tpData.traceData.transmitterID];

		const VCT::IntersectableEntity& ie = data.sceneData.intersectableEntities[ieID];
		Ray ray = Ray(tx.position, ie.rtPoint);
		if (IsValidInteraction(ie.type, tpData, data.coneTracingData) && ray.Trace(data.sceneData.rtParams, ieID, ie.type))
		{
			bool interactionWritten = HandleValidInteraction(ray, ie, tpData);
			data.transmitIndexProcessed[ieID] = interactionWritten;

			if (!interactionWritten)
				*data.status = VCT::Status::ProcessingRequired;
		}
	}
}

extern "C" __global__ void __miss__Refine()
{
	OnMiss();
}

extern "C" __global__ void __intersection__Refine()
{
	OnIntersect(data.sceneData.refineRtParams);
}

extern "C" __global__ void __closesthit__Refine()
{
	glm::vec3 rayOrigin = reinterpret_cast<glm::vec3&>(optixGetWorldRayOrigin());
	glm::vec3 rayDirection = reinterpret_cast<glm::vec3&>(optixGetWorldRayDirection());
	float distance = __uint_as_float(optixGetAttribute_0());
	uint32_t primitiveID = optixGetPrimitiveIndex();
	uint32_t primitivePointID = VCT::Constants::InvalidPointIndex;
	glm::vec3 normal = RefineNormal(rayOrigin, rayDirection, distance, primitiveID, data.subIePrimitiveNeighbors[primitiveID], data.sceneData.refineRtParams, primitivePointID);
	optixSetPayload_0(data.sceneData.refineRtParams.primitiveInfos[optixGetPrimitiveIndex()].ID);
	optixSetPayload_2(optixGetAttribute_0());
	optixSetPayload_3(__float_as_uint(normal.x));
	optixSetPayload_4(__float_as_uint(normal.y));
	optixSetPayload_5(__float_as_uint(normal.z));
	optixSetPayload_6(primitivePointID);
}

inline __device__ void FinalizePath(VCT::TraceData& result)
{
	float timeDelay = 0.0f;
	const VCT::Transmitter& tx = data.sceneData.transmitters[result.transmitterID];
	const VCT::Receiver& rx = data.sceneData.receivers[result.receiverID];

	glm::vec3 prevPos = tx.position;
	for (uint32_t i = 0; i < result.numInteractions; ++i)
	{
		timeDelay += glm::length(prevPos - result.interactions[i].position) * VCT::Constants::InvLightSpeedInVacuum;
		prevPos = result.interactions[i].position;
	}
	timeDelay += glm::length(prevPos - rx.position) * VCT::Constants::InvLightSpeedInVacuum;
	result.timeDelay = timeDelay;
}

extern "C" __global__ void __raygen__Refine()
{
	const VCT::TraceData& originalPath = data.pathsToRefine[optixGetLaunchIndex().x];
	VCT::TraceData resultPath = originalPath;
	if (PathRefiner(data.sceneData, data.coneTracingData, originalPath).Refine(data.refineParams, resultPath))
	{
		FinalizePath(resultPath);
		data.refinedPaths[atomicAdd(data.numRefinedPaths, 1)] = resultPath;
	}
}