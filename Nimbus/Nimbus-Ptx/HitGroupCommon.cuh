#pragma once
#include "SDF.cuh"
#include "Nimbus/Utils.hpp"
#include "Ray.cuh"

inline __device__ glm::vec3 GetWorldRayOrigin()
{
	float3 origin = optixGetWorldRayOrigin();
	return glm::vec3(origin.x, origin.y, origin.z);
}

inline __device__ glm::vec3 GetWorldRayDirection()
{
	float3 direction = optixGetWorldRayDirection();
	return glm::vec3(direction.x, direction.y, direction.z);
}

inline __device__ glm::vec3 GetObjectRayOrigin()
{
	float3 origin = optixGetObjectRayOrigin();
	return glm::vec3(origin.x, origin.y, origin.z);
}

inline __device__ glm::vec3 GetObjectRayDirection()
{
	float3 direction = optixGetObjectRayDirection();
	return glm::vec3(direction.x, direction.y, direction.z);
}

inline __device__ glm::vec2 GetTriangleBarycentrics()
{
	float2 barys = optixGetTriangleBarycentrics();
	return glm::vec2(barys.x, barys.y);
}

inline __device__ glm::vec3 GetNormalForTriangle(uint32_t primitiveIndex, const Nimbus::EnvironmentData& env)
{
	if (env.triangle.useFaceNormals)
		return env.triangle.faces[primitiveIndex].normal;

	glm::vec2 barys = GetTriangleBarycentrics();
	const uint32_t* indices = env.triangle.indices;
	const glm::vec3& n0 = env.triangle.normals[indices[primitiveIndex * 3 + 0]];
	const glm::vec3& n1 = env.triangle.normals[indices[primitiveIndex * 3 + 1]];
	const glm::vec3& n2 = env.triangle.normals[indices[primitiveIndex * 3 + 2]];
	return glm::normalize((1.0f - barys.x - barys.y) * n0 + barys.x * n1 + barys.y * n2);
}

inline __device__ void OnClosestHit(const Nimbus::EnvironmentData& env)
{
	const Nimbus::PrimitivePoint& pp = env.pc.primitivePoints[optixGetAttribute_3()];
	Ray::Payload* payload = Nimbus::Utils::UnpackPointer32<Ray::Payload>(optixGetPayload_0(), optixGetPayload_1());
	payload->t = optixGetRayTmax();
	payload->label = pp.label;
	payload->material = pp.materialID;
	payload->normal = glm::vec3(__uint_as_float(optixGetAttribute_0()), __uint_as_float(optixGetAttribute_1()), __uint_as_float(optixGetAttribute_2()));
	payload->rtPointIndex = optixGetPrimitiveIndex();
}

inline __device__ void OnClosestHitTriangle(const Nimbus::EnvironmentData& env)
{
	float t = optixGetRayTmax();
	uint32_t primitiveIndex = optixGetPrimitiveIndex();
	const Nimbus::Face& face = env.triangle.faces[primitiveIndex];
	glm::vec3 rayOrigin = GetWorldRayOrigin();
	glm::vec3 rayDir = GetWorldRayDirection();
	glm::vec3 hitPoint = rayOrigin + rayDir * t;
	glm::vec3 hitPointBiased = rayOrigin + rayDir * (t + 1e-4f);
	glm::vec3 hitPointBiased2 = rayOrigin + rayDir * (t - 1e-4f);
	
	uint32_t rtPointIndex0 = env.triangle.voxelToRtPointIndexMap[Nimbus::Utils::WorldToVoxelID(hitPoint, env.vwInfo)];
	uint32_t rtPointIndex1 = env.triangle.voxelToRtPointIndexMap[Nimbus::Utils::WorldToVoxelID(hitPointBiased, env.vwInfo)];
	uint32_t rtPointIndex2 = env.triangle.voxelToRtPointIndexMap[Nimbus::Utils::WorldToVoxelID(hitPointBiased2, env.vwInfo)];
	
	uint32_t rtPointIndex = rtPointIndex0 * (rtPointIndex0 != Nimbus::Constants::InvalidPointIndex)
						  + rtPointIndex1 * ((rtPointIndex0 == Nimbus::Constants::InvalidPointIndex) & (rtPointIndex1 != Nimbus::Constants::InvalidPointIndex))
						  + rtPointIndex2 * ((rtPointIndex0 == Nimbus::Constants::InvalidPointIndex) & (rtPointIndex1 == Nimbus::Constants::InvalidPointIndex));
	
	auto c = Nimbus::Utils::GetVoxelCoord(hitPoint, env.vwInfo);
	auto c2 = Nimbus::Utils::GetVoxelCoord(hitPointBiased2, env.vwInfo);

	Ray::Payload* payload = Nimbus::Utils::UnpackPointer32<Ray::Payload>(optixGetPayload_0(), optixGetPayload_1());
	payload->t = t * static_cast<float>(rtPointIndex != Nimbus::Constants::InvalidPointIndex);
	payload->label = face.label;
	payload->material = face.material;
	payload->normal = Nimbus::Utils::FixNormal(rayDir, GetNormalForTriangle(primitiveIndex, env));
	payload->rtPointIndex = rtPointIndex;
}

inline __device__ void OnClosestHitRIS(const Nimbus::EnvironmentData& env)
{
	uint32_t risIndex = optixGetPrimitiveIndex() / 2u;
	Ray::Payload* payload = Nimbus::Utils::UnpackPointer32<Ray::Payload>(optixGetPayload_0(), optixGetPayload_1());
	uint32_t objectId = env.ris.objectIds[risIndex];
	glm::vec3 normal = env.ris.normals[risIndex];

	payload->t = optixGetRayTmax();
	payload->label = objectId;
	payload->material = Nimbus::Constants::InvalidPointIndex;
	payload->normal = normal;
	payload->rtPointIndex = Nimbus::Constants::RisHit;
}

inline __device__ bool OnIntersect(const Nimbus::RayTracingParams& rtParams)
{
	glm::vec3 rayOrigin = GetObjectRayOrigin();
	glm::vec3 rayDirection = GetObjectRayDirection();
	uint32_t primitiveIndex = optixGetPrimitiveIndex();
	const Nimbus::IEPrimitiveInfo& ieInfo = rtParams.env.pc.primitiveInfos[optixGetPrimitiveIndex()];

	const OptixAabb& aabb = rtParams.env.pc.primitives[primitiveIndex];
	glm::vec3 rtPoint = glm::vec3(aabb.minX + aabb.maxX, aabb.minY + aabb.maxY, aabb.minZ + aabb.maxZ) * 0.5f;

	float distance = 0.0f;
	glm::vec3 normal = glm::vec3(0.0f);
	uint32_t ppIdx = Nimbus::Constants::InvalidPointIndex;
	bool implicitIntersect = IntersectWithImplicitSurface(rayOrigin, rayDirection, rtPoint, ieInfo, rtParams, distance, normal, ppIdx);
	implicitIntersect &= Nimbus::Utils::IsPointInAabb(aabb, rayOrigin + rayDirection * distance, rtParams.sampleRadius);
	if (implicitIntersect)
		optixReportIntersection(distance, 0, __float_as_uint(normal.x), __float_as_uint(normal.y), __float_as_uint(normal.z), ppIdx);

	return implicitIntersect;
}