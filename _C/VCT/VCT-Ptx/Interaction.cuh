#pragma once
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "Types.hpp"
#include "Utils.hpp"
#include "Ray.cuh"
#include "Propagation.cuh"

extern __constant__ VCT::VCTData data;

inline __device__ bool IsCriticalAngle(const glm::vec3& v)
{
	return isnan(v.x);
}

inline __device__ bool HandleReceiverInteraction(const Ray& ray, const VCT::IntersectableEntity& ie, const VCT::TraceProcessingData& tpData)
{
	uint32_t recvPathIndex = atomicAdd(data.pathData.numReceivedPaths, 1);
	bool allocSuccess = recvPathIndex < data.pathData.maxNumReceivedPaths;

	if (allocSuccess)
	{
		VCT::TraceData& result = data.pathData.coarsePaths[*data.pathData.activeBufferIndex][recvPathIndex];
		float dist = glm::length(ray.GetOrigin() - data.sceneData.receivers[ie.receiverID].position);
		result = tpData.traceData;
		result.timeDelay += dist * VCT::Constants::InvLightSpeedInVacuum;
		result.receiverID = ie.receiverID;
	}
	return allocSuccess;
}

inline __device__ bool HandleEdgeInteraction(const Ray& ray, const VCT::IntersectableEntity& ie, const VCT::TraceProcessingData& tpData)
{
	const VCT::DiffractionEdgeSegment& edgeSegment = data.coneTracingData.diffractionEdgeSegments[ie.edgeSegmentID];
	const VCT::DiffractionEdge& edge = data.coneTracingData.diffractionEdges[edgeSegment.parentID];

	glm::vec3 dir = ray.GetDirection();

	if (!edge.IsValidIncidentRayForDiffraction(dir))
		return true;

	glm::vec3 reflectedRay = glm::reflect(dir, edge.right);

	float cosAngle = glm::dot(reflectedRay, edge.forward);
	float sinAngle = glm::sqrt(1 - cosAngle * cosAngle);

	uint32_t index = edge.firstInfoIndex + static_cast<uint32_t>(std::abs(sinAngle) * VCT::Constants::UnitCircleDiscretizationCount);
	glm::vec3 center = ie.rtPoint;

	glm::mat4 mat = glm::mat4(glm::vec4(edge.right, 0.0f),
							  glm::vec4(edge.forward, 0.0f),
							  glm::vec4(edge.up, 0.0f),
							  glm::vec4(center + edge.forward * cosAngle, 1.0f));

	uint32_t iaIndex = tpData.traceData.numInteractions;

	const VCT::IndexInfo& diffIndexInfo = data.coneTracingData.diffractionIndexInfos[index];
	uint32_t firstRay = atomicAdd(&data.pathData.pathProcessingData->numPaths, diffIndexInfo.count);
	bool allocSuccess = firstRay + diffIndexInfo.count <= data.pathData.maxNumPropPaths[iaIndex];

	if (allocSuccess)
	{
		atomicAdd(&data.pathData.pathProcessingData->nextNumPathsToProcess, diffIndexInfo.count);

		uint32_t localRayIndex = 0;
		for (uint32_t rayIndex = 0; rayIndex < diffIndexInfo.count; ++rayIndex)
		{
			const VCT::DiffractionRay& diffractionRay = data.coneTracingData.diffractionRays[diffIndexInfo.first + rayIndex];

			VCT::PropagationData& propData = data.pathData.propPaths[iaIndex][firstRay + localRayIndex++];

			glm::vec4 translation = mat * glm::vec4(diffractionRay.direction.x * sinAngle, 0.0f, diffractionRay.direction.y * sinAngle, 1.0f);

			glm::vec3 planeDir0 = glm::vec3(mat * glm::vec4(diffractionRay.planeDirections[0].x * sinAngle, 0.0f, diffractionRay.planeDirections[0].y * sinAngle, 1.0f)) - center;
			glm::vec3 planeDir1 = glm::vec3(mat * glm::vec4(diffractionRay.planeDirections[1].x * sinAngle, 0.0f, diffractionRay.planeDirections[1].y * sinAngle, 1.0f)) - center;

			glm::vec3 planeNormal0 = glm::normalize(glm::cross(planeDir0, edge.forward));
			glm::vec3 planeNormal1 = glm::normalize(glm::cross(planeDir1, -edge.forward));


			propData.tpData = tpData;
			propData.tpData.incidentIor = 1.0f;
			propData.tpData.numDiffractions++;

			propData.tpData.traceData.numInteractions++;
			float dist = glm::length(tpData.traceData.interactions[tpData.traceData.numInteractions - 1].position - ray.GetOrigin());
			propData.tpData.traceData.timeDelay += dist * VCT::Constants::InvLightSpeedInVacuum;

			propData.tpData.traceData.interactions[iaIndex].position = center;
			propData.tpData.traceData.interactions[iaIndex].ieID = ray.GetPayload().hitIeID;
			propData.tpData.traceData.interactions[iaIndex].label = data.coneTracingData.diffractionEdgeSegments[data.sceneData.intersectableEntities[ray.GetPayload().hitIeID].edgeSegmentID].parentID;
			propData.tpData.traceData.interactions[iaIndex].normal = VCT::Utils::FixNormal(ray.GetDirection(), edge.right);
			propData.tpData.traceData.interactions[iaIndex].type = VCT::InteractionType::Diffraction;

			propData.voxelTraceData.finished = false;
			propData.voxelTraceData.rayDirection = glm::normalize(glm::vec3(translation) - center);
			propData.voxelTraceData.voxel = VCT::Utils::WorldToVoxel(propData.tpData.traceData.interactions[iaIndex].position, data.coneTracingData.voxelWorldInfo);
			propData.voxelTraceData.localIeID = 0;
			propData.voxelTraceData.localVoxel = glm::u16vec3(0);

			propData.voxelTraceData.previousVoxel = propData.voxelTraceData.voxel;
			InitVoxelHistory(propData.voxelTraceData.voxelHistory);

			propData.voxelTraceData.intersectionData.rayCone = Cone(propData.voxelTraceData.voxel, propData.voxelTraceData.rayDirection, data.coneTracingData.cosDiffuseAngle, data.coneTracingData.sinDiffuseAngle);
			propData.voxelTraceData.intersectionData.separationPlanes[0] = Plane(propData.voxelTraceData.voxel, planeNormal0);
			propData.voxelTraceData.intersectionData.separationPlanes[1] = Plane(propData.voxelTraceData.voxel, planeNormal1);
		}
	}

	return allocSuccess;
}

inline __device__ bool HandleReflectionInteraction(const Ray& ray, const VCT::TraceProcessingData& tpData, VCT::PropagationData& result)
{
	uint32_t iaIndex = tpData.traceData.numInteractions;
	glm::vec3 surfaceNormal = ray.GetPayload().GetNormal();
	glm::vec3 reflectedRay = glm::reflect(ray.GetDirection(), surfaceNormal);

	result.tpData = tpData;
	result.tpData.incidentIor = 1.0f;

	result.tpData.traceData.numInteractions++;
	result.tpData.traceData.timeDelay += ray.GetPayload().GetDistance() * VCT::Constants::InvLightSpeedInVacuum;
	result.tpData.traceData.interactions[iaIndex].position = ray.GetOrigin() + ray.GetDirection() * ray.GetPayload().GetDistance();
	result.tpData.traceData.interactions[iaIndex].ieID = ray.GetPayload().hitIeID;
	result.tpData.traceData.interactions[iaIndex].label = data.sceneData.intersectableEntities[ray.GetPayload().hitIeID].surfaceLabel;
	result.tpData.traceData.interactions[iaIndex].normal = surfaceNormal;
	result.tpData.traceData.interactions[iaIndex].type = VCT::InteractionType::Reflection;
	
	result.voxelTraceData.finished = false;
	result.voxelTraceData.rayDirection = reflectedRay;
	result.voxelTraceData.voxel = VCT::Utils::WorldToVoxel(result.tpData.traceData.interactions[iaIndex].position, data.coneTracingData.voxelWorldInfo);
	result.voxelTraceData.localIeID = 0;
	result.voxelTraceData.localVoxel = glm::u16vec3(0);
	result.voxelTraceData.previousVoxel = result.voxelTraceData.voxel;
	InitVoxelHistory(result.voxelTraceData.voxelHistory);

	result.voxelTraceData.intersectionData.rayCone = Cone(result.voxelTraceData.voxel, result.voxelTraceData.rayDirection, data.coneTracingData.reflCosDiffuseAngle, data.coneTracingData.reflSinDiffuseAngle);
	result.voxelTraceData.intersectionData.separationPlanes[0] = Plane(result.voxelTraceData.voxel + surfaceNormal * VCT::Constants::SeparationPlaneBias, -surfaceNormal);
	result.voxelTraceData.intersectionData.separationPlanes[1] = Plane(result.voxelTraceData.voxel + surfaceNormal * VCT::Constants::SeparationPlaneBias, -surfaceNormal);

	return true;
}

inline __device__ bool HandleSurfaceInteraction(const Ray& ray, const VCT::TraceProcessingData& tpData)
{
	VCT::PropagationData reflectionResult{}, refractionResult{};
	HandleReflectionInteraction(ray, tpData, reflectionResult);
	uint32_t iaIndex = tpData.traceData.numInteractions;
	bool allocSuccess = true;
	constexpr uint32_t allocCount = 1u;
	uint32_t propIndex = atomicAdd(&data.pathData.pathProcessingData->numPaths, allocCount);
	allocSuccess = propIndex + allocCount <= data.pathData.maxNumPropPaths[iaIndex];
	if (allocSuccess)
		atomicAdd(&data.pathData.pathProcessingData->nextNumPathsToProcess, allocCount);

	if (allocSuccess)
		data.pathData.propPaths[iaIndex][propIndex] = reflectionResult;

	return allocSuccess;
}

inline __device__ bool HandleValidInteraction(const Ray& ray, const VCT::IntersectableEntity& ie, const VCT::TraceProcessingData& tpData)
{
	switch (ie.type) // Could be callable
	{
	case VCT::IEType::Receiver: return HandleReceiverInteraction(ray, ie, tpData);
	case VCT::IEType::Edge:		return HandleEdgeInteraction(ray, ie, tpData);
	case VCT::IEType::Surface:  return HandleSurfaceInteraction(ray, tpData);
	default:					return true;
	}
}