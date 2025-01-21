#pragma once
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "Nimbus/Types.hpp"
#include "Nimbus/Utils.hpp"

inline __device__ bool Sign(float value)
{
	return value >= 0.0f;
}

inline __device__ float GaussianWeight(float distanceSq, float varianceSq)
{
	return glm::exp(-(distanceSq / (2 * varianceSq)));
}

inline __device__ void Nx(const glm::vec3& position,
						  const Nimbus::IEPrimitiveInfo& primitiveInfo,
						  const Nimbus::PrimitivePoint* surfacePoints,
						  float varianceSq,
						  float& denominator,
						  glm::vec3& resultNormal,
						  float& closestPointDistSq,
						  uint32_t& primitivePointID)
{
	for (uint32_t localPointIndex = 0; localPointIndex < primitiveInfo.pointIndexInfo.count; ++localPointIndex)
	{
		const Nimbus::PrimitivePoint& primitive = surfacePoints[primitiveInfo.pointIndexInfo.first + localPointIndex];
		glm::vec3 diff = primitive.position - position;
		float distanceSq = glm::dot(diff, diff);
		primitivePointID = distanceSq < closestPointDistSq ? primitiveInfo.pointIndexInfo.first + localPointIndex : primitivePointID;
		closestPointDistSq = distanceSq < closestPointDistSq ? distanceSq : closestPointDistSq;
		float weight = GaussianWeight(distanceSq, varianceSq);
		denominator += weight;
		resultNormal += weight * primitive.normal;
	}
}

inline __device__ float ComputeSdf(const Nimbus::PrimitivePoint* surfacePoints,
								   const Nimbus::IEPrimitiveInfo& primitiveInfo,
								   const glm::vec3& position,
								   float varianceSq,
								   glm::vec3& resultNormal,
								   uint32_t& primitivePointID)
{
	float closestPointDistSq = 5000.0f;
	float denominator = 0.0f;
	glm::vec3 p = glm::vec3(0.0f);
	resultNormal = glm::vec3(0.0f);

	for (uint32_t localPointIndex = 0; localPointIndex < primitiveInfo.pointIndexInfo.count; ++localPointIndex)
	{
		const Nimbus::PrimitivePoint& primitive = surfacePoints[primitiveInfo.pointIndexInfo.first + localPointIndex];
		glm::vec3 diff = primitive.position - position;
		float distanceSq = glm::dot(diff, diff);
		primitivePointID = distanceSq < closestPointDistSq ? primitiveInfo.pointIndexInfo.first + localPointIndex : primitivePointID;
		closestPointDistSq = distanceSq < closestPointDistSq ? distanceSq : closestPointDistSq;

		float weight = GaussianWeight(distanceSq, varianceSq);
		denominator += weight;
		p += weight * primitive.position;
		resultNormal += weight * primitive.normal;
	}
	p /= denominator;
	resultNormal /= denominator;
	return glm::dot(position - p, resultNormal);
}

inline __device__ bool Sdf(const Nimbus::PrimitivePoint* surfacePoints,
							const Nimbus::IEPrimitiveInfo& primitiveInfo,
							const glm::vec3& initialPos,
							const glm::vec3& rayDir,
							float sampleDistance,
							float sampleRadius,
							float sdfThreshold,
							float varianceSq,
							glm::vec3& resultPos,
							glm::vec3& resultNormal,
							float& resultSdf,
							uint32_t& resultPrimitivePointIndex)
{
	glm::vec3 normal = {};
	uint32_t ppID = Nimbus::Constants::InvalidPointIndex;
	float sdf = ComputeSdf(surfacePoints, primitiveInfo, initialPos, varianceSq, normal, ppID);
	glm::vec3 p = initialPos;
	if (!glm::isnan(sdf) && glm::abs(sdf) < sdfThreshold)
	{
		resultPos = p;
		resultNormal = normal;
		resultSdf = sdf;
		resultPrimitivePointIndex = ppID;
		return true;
	}
	float travelDist = 0.0f;
	while (travelDist < sampleDistance)
	{
		float dist = glm::isnan(sdf) ? sampleRadius : glm::abs(sdf);
		travelDist += dist;
		glm::vec3 cp = p + rayDir * dist;
		glm::vec3 cNormal = glm::vec3(0.0f);
		uint32_t cPpID = Nimbus::Constants::InvalidPointIndex;
		float cSdf = ComputeSdf(surfacePoints, primitiveInfo, cp, varianceSq, cNormal, cPpID);
		bool validIntersection = (Sign(sdf) != Sign(cSdf)) && !glm::isnan(sdf) && !glm::isnan(cSdf);

		if (validIntersection)
		{
			resultPos = p;
			resultNormal = normal;
			resultSdf = sdf;
			resultPrimitivePointIndex = ppID;
			return true;
		}

		if (!glm::isnan(cSdf) && glm::abs(cSdf) < sdfThreshold)
		{
			resultPos = cp;
			resultNormal = cNormal;
			resultSdf = cSdf;
			resultPrimitivePointIndex = cPpID;
			return true;
		}

		sdf = cSdf;
		p = cp;
	}
	return false;
}

inline __device__ glm::vec3 RayPlaneIntersect(const glm::vec3 origin, const glm::vec3& direction, const glm::vec3& planePoint, const glm::vec3& planeNormal, bool& found)
{
	float denom = glm::dot(planeNormal, direction);
	found &= glm::abs(denom) > 1e-4f;
	return origin + glm::dot(planeNormal, planePoint - origin) / denom * direction;
}

inline __device__ bool IntersectWithImplicitSurface(const glm::vec3& rayOrigin,
													const glm::vec3& rayDirection,
													const glm::vec3& rayDestination,
													const Nimbus::IEPrimitiveInfo& primitiveInfo,
													const Nimbus::RayTracingParams& rtParams,
													float& resultDistance,
													glm::vec3& resultNormal,
													uint32_t& resultPrimitivePointIndex)
{
	float variance = rtParams.sampleRadius * rtParams.varianceFactor;
	float varianceSq = variance * variance;

	glm::vec3 closestPoint = Nimbus::Utils::ProjectPointToRay(rayOrigin, rayDirection, rayDestination);
	float t0 = glm::length(rayOrigin - (closestPoint - rayDirection * (rtParams.sampleDistance * 0.5f)));

	glm::vec3 resPos;
	glm::vec3 resNorm;
	float sdf;
	bool found = Sdf(rtParams.env.pc.primitivePoints, primitiveInfo, rayOrigin + rayDirection * t0, rayDirection, rtParams.sampleDistance, rtParams.sampleRadius, rtParams.sdfThreshold, varianceSq, resPos, resNorm, sdf, resultPrimitivePointIndex);
	glm::vec3 normal = Nimbus::Utils::FixNormal(rayDirection, glm::normalize(resNorm));
	glm::vec3 planePoint = resPos + glm::abs(sdf) * -normal;
	resultDistance = glm::length(RayPlaneIntersect(rayOrigin, rayDirection, planePoint, normal, found) - rayOrigin);
	resultNormal = normal;
	return found;
}
