#pragma once
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "Nimbus/Types.hpp"
#include "Nimbus/Utils.hpp"

inline __device__ float EvaluateGaussianDist(const Nimbus::PrimitivePoint& pp,
	float strength,
	const float dist,
	float lambdaZ)
{
	float gauss = -0.5f * strength;
	return glm::exp(gauss - lambdaZ * dist);
}

inline __device__ float RayPlaneIntersectDist(const glm::vec3 origin, const glm::vec3& direction, const glm::vec3& planePoint, const glm::vec3& planeNormal, bool& found)
{
	float denom = glm::dot(planeNormal, direction);
	found &= glm::abs(denom) > 1e-4f;
	return glm::dot(planeNormal, planePoint - origin) / denom;
}

inline __device__ float FindTmin(const glm::vec3& rayOrigin, const glm::vec3& rayDirection, const Nimbus::PrimitivePoint* surfacePoints, const Nimbus::IEPrimitiveInfo& primitiveInfo, float pointRadius, uint32_t& ppIndex)
{
	float tmin = FLT_MAX;

	for (uint32_t localPointIndex = 0; localPointIndex < primitiveInfo.pointIndexInfo.count; ++localPointIndex)
	{
		uint32_t primitiveIndex = primitiveInfo.pointIndexInfo.first + localPointIndex;
		const Nimbus::PrimitivePoint& primitive = surfacePoints[primitiveIndex];
		bool intersect = true;
		float t = RayPlaneIntersectDist(rayOrigin, rayDirection, primitive.position, primitive.normal, intersect);
		glm::vec3 i = rayOrigin + rayDirection * t;
		glm::vec3 diffiu = i - primitive.position;
		float iuDistSq = glm::dot(diffiu, diffiu);
		float gaussStrength = glm::dot(diffiu, diffiu) / (pointRadius * pointRadius);

		if (!intersect || gaussStrength > 1.0f || t < 0.0f)
			continue;

		tmin = glm::min(tmin, t);
		ppIndex = tmin == t ? primitiveIndex : ppIndex;
	}
	return tmin;
}

inline __device__ void FindIntersection(const glm::vec3& rayOrigin, const glm::vec3& rayDirection, const Nimbus::PrimitivePoint* surfacePoints, const Nimbus::IEPrimitiveInfo& primitiveInfo,
	float pointRadius, float tmin, float lambdaDistance, glm::vec3& nAccum, float& tAccum, float& wAccum)
{
	for (uint32_t localPointIndex = 0; localPointIndex < primitiveInfo.pointIndexInfo.count; ++localPointIndex)
	{
		const Nimbus::PrimitivePoint& primitive = surfacePoints[primitiveInfo.pointIndexInfo.first + localPointIndex];
		bool intersect = true;
		float t = RayPlaneIntersectDist(rayOrigin, rayDirection, primitive.position, primitive.normal, intersect);
		glm::vec3 i = rayOrigin + rayDirection * t;
		glm::vec3 diffiu = i - primitive.position;
		float iuDistSq = glm::dot(diffiu, diffiu);
		float gaussStrength = glm::dot(diffiu, diffiu) / (pointRadius * pointRadius);

		if (!intersect || gaussStrength > 1.0f)
			continue;

		float weight = EvaluateGaussianDist(primitive, gaussStrength, t - tmin, lambdaDistance);
		nAccum += Nimbus::Utils::FixNormal(rayDirection, primitive.normal) * weight;
		tAccum += t * weight;
		wAccum += weight;
	}
}