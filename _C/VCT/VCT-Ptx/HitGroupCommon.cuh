#pragma once
#include "SDF.cuh"
#include "Utils.hpp"

inline __device__ void OnClosestHit(uint32_t hitID)
{
	optixSetPayload_0(hitID);
	optixSetPayload_2(optixGetAttribute_0());
	optixSetPayload_3(optixGetAttribute_1());
	optixSetPayload_4(optixGetAttribute_2());
	optixSetPayload_5(optixGetAttribute_3());
	optixSetPayload_6(optixGetPrimitiveIndex());
}

inline __device__ void OnMiss()
{
	optixSetPayload_0(optixGetPayload_1() != static_cast<uint32_t>(VCT::IEType::Surface) ? optixGetPayload_0() : VCT::Constants::InvalidPointIndex);
}

inline __device__ bool OnIntersect(const VCT::RayTracingParams& rtParams)
{
	glm::vec3 rayOrigin = reinterpret_cast<glm::vec3&>(optixGetObjectRayOrigin());
	glm::vec3 rayDirection = reinterpret_cast<glm::vec3&>(optixGetObjectRayDirection());
	uint32_t primitiveIndex = optixGetPrimitiveIndex();
	const VCT::IEPrimitiveInfo& ieInfo = rtParams.primitiveInfos[optixGetPrimitiveIndex()];

	const OptixAabb& aabb = rtParams.primitives[primitiveIndex];
	glm::vec3 rtPoint = glm::vec3(aabb.minX + aabb.maxX, aabb.minY + aabb.maxY, aabb.minZ + aabb.maxZ) * 0.5f;

	float distance = 0.0f;
	glm::vec3 normal = glm::vec3(0.0f);
	bool implicitIntersect = IntersectWithImplicitSurface(rayOrigin, rayDirection, rtPoint, ieInfo, rtParams, distance, normal);
	implicitIntersect &= VCT::Utils::IsPointInAabb(aabb, rayOrigin + rayDirection * distance, rtParams.sampleRadius);

	if (implicitIntersect)
		optixReportIntersection(distance, 0, __float_as_uint(distance), __float_as_uint(normal.x), __float_as_uint(normal.y), __float_as_uint(normal.z));

	return implicitIntersect;
}