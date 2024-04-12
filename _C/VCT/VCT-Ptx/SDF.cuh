#pragma once
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "Types.hpp"
#include "Utils.hpp"

inline __device__ bool Sign(float value)
{
	return value >= 0.0f;
}

inline __device__ float GaussianWeight(float distanceSq, float varianceSq)
{
	return glm::exp(-(distanceSq / (2 * varianceSq)));
}

inline __device__ void Nx(const glm::vec3& position,
						  const VCT::IEPrimitiveInfo& primitiveInfo,
						  const VCT::PrimitivePoint* surfacePoints,
						  float varianceSq,
						  float& denominator,
						  glm::vec3& resultNormal,
						  float& closestPointDistSq,
						  uint32_t& primitivePointID)
{
	for (uint32_t localPointIndex = 0; localPointIndex < primitiveInfo.pointIndexInfo.count; ++localPointIndex)
	{
		const VCT::PrimitivePoint& primitive = surfacePoints[primitiveInfo.pointIndexInfo.first + localPointIndex];
		glm::vec3 diff = primitive.position - position;
		float distanceSq = glm::dot(diff, diff);
		primitivePointID = distanceSq < closestPointDistSq ? primitiveInfo.pointIndexInfo.first + localPointIndex : primitivePointID;
		closestPointDistSq = distanceSq < closestPointDistSq ? distanceSq : closestPointDistSq;
		float weight = GaussianWeight(distanceSq, varianceSq);
		denominator += weight;
		resultNormal += weight * primitive.normal;
	}
}

inline __device__ glm::vec3 RefineNormal(const glm::vec3& rayOrigin,
										 const glm::vec3& rayDirection,
										 float distance,
										 uint32_t primitiveID,
										 const VCT::PrimitiveNeighbors& primitiveNeighbors,
										 const VCT::RayTracingParams& rtParams,
										 uint32_t& primitivePointID)
{
	float denom = 0.0f;
	glm::vec3 resultNormal = glm::vec3(0.0f);
	float closestPointDistSq = rtParams.sampleDistance * rtParams.sampleDistance;
	float variance = rtParams.sampleRadius * rtParams.varianceFactor;
	float varianceSq = variance * variance;
	glm::vec3 position = rayOrigin + rayDirection * distance;
	Nx(position, rtParams.primitiveInfos[primitiveID], rtParams.primitivePoints, varianceSq, denom, resultNormal, closestPointDistSq, primitivePointID);
	
	for (uint32_t i = 0; i < primitiveNeighbors.count; ++i)
		Nx(position, rtParams.primitiveInfos[primitiveNeighbors.neighbors[i]], rtParams.primitivePoints, varianceSq, denom, resultNormal, closestPointDistSq, primitivePointID);
	
	return glm::vec3(VCT::Utils::FixNormal(rayDirection, glm::normalize(resultNormal / denom)));
}

inline __device__ float SDF2GetSdf(const VCT::PrimitivePoint* surfacePoints, const VCT::IEPrimitiveInfo& primitiveInfo, const glm::vec3& position, float varianceSq, glm::vec3& resultNormal)
{
	float denominator = 0.0f;
	glm::vec3 p = glm::vec3(0.0f);
	resultNormal = glm::vec3(0.0f);

	for (uint32_t localPointIndex = 0; localPointIndex < primitiveInfo.pointIndexInfo.count; ++localPointIndex)
	{
		const VCT::PrimitivePoint& primitive = surfacePoints[primitiveInfo.pointIndexInfo.first + localPointIndex];
		glm::vec3 diff = primitive.position - position;
		float distanceSq = glm::dot(diff, diff);
		float weight = GaussianWeight(distanceSq, varianceSq);
		denominator += weight;
		p += weight * primitive.position;
		resultNormal += weight * primitive.normal;
	}
	p /= denominator;
	resultNormal /= denominator;
	return glm::dot(position - p, resultNormal);
}

inline __device__ bool SDF2(const VCT::PrimitivePoint* surfacePoints,
							const VCT::IEPrimitiveInfo& primitiveInfo,
							const glm::vec3& initialPos,
							const glm::vec3& rayDir,
							float sampleDistance,
							float sampleRadius,
							float sdfThreshold,
							float varianceSq,
							glm::vec3& resultPos,
							glm::vec3& resultNormal,
							float& resultSdf)
{
	glm::vec3 normal = {};
	float sdf = SDF2GetSdf(surfacePoints, primitiveInfo, initialPos, varianceSq, normal);
	glm::vec3 p = initialPos;
	if (!glm::isnan(sdf) && glm::abs(sdf) < sdfThreshold)
	{
		resultPos = p;
		resultNormal = normal;
		resultSdf = sdf;
		return true;
	}
	float travelDist = 0.0f;
	while (travelDist < sampleDistance)
	{
		float dist = glm::isnan(sdf) ? sampleRadius : glm::abs(sdf);
		travelDist += dist;
		glm::vec3 cp = p + rayDir * dist;
		glm::vec3 cNormal = glm::vec3(0.0f);
		float cSdf = SDF2GetSdf(surfacePoints, primitiveInfo, cp, varianceSq, cNormal);
		bool validIntersection = (Sign(sdf) != Sign(cSdf)) && !glm::isnan(sdf) && !glm::isnan(cSdf);

		if (validIntersection)
		{
			resultPos = p;
			resultNormal = normal;
			resultSdf = sdf;
			return true;
		}

		if (!glm::isnan(cSdf) && glm::abs(cSdf) < sdfThreshold)
		{
			resultPos = cp;
			resultNormal = cNormal;
			resultSdf = cSdf;
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
	found = glm::abs(denom) > 1e-4f;
	return origin + glm::dot(planeNormal, planePoint - origin) / denom * direction;
}

inline __device__ bool IntersectWithImplicitSurface(const glm::vec3& rayOrigin,
													const glm::vec3& rayDirection,
													const glm::vec3& rayDestination,
													const VCT::IEPrimitiveInfo& primitiveInfo,
													const VCT::RayTracingParams& rtParams,
													float& resultDistance,
													glm::vec3& resultNormal)
{
	float variance = rtParams.sampleRadius * rtParams.varianceFactor;
	float varianceSq = variance * variance;

	glm::vec3 closestPoint = VCT::Utils::ProjectPointToRay(rayOrigin, rayDirection, rayDestination);
	float t0 = glm::length(rayOrigin - (closestPoint - rayDirection * (rtParams.sampleDistance * 0.5f)));

	glm::vec3 resPos;
	glm::vec3 resNorm;
	float sdf;
	bool found = SDF2(rtParams.primitivePoints, primitiveInfo, rayOrigin + rayDirection * t0, rayDirection, rtParams.sampleDistance, rtParams.sampleRadius, rtParams.sdfThreshold, varianceSq, resPos, resNorm, sdf);
	glm::vec3 normal = VCT::Utils::FixNormal(rayDirection, glm::normalize(resNorm));
	glm::vec3 planePoint = resPos + glm::abs(sdf) * -normal;
	resultDistance = glm::length(RayPlaneIntersect(rayOrigin, rayDirection, planePoint, normal, found) - rayOrigin);
	resultNormal = normal;
	return found;
}
