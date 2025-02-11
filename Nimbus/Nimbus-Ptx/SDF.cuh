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

inline __device__ float EvaluateGaussian(const Nimbus::PrimitivePoint& pp,
									     const glm::vec3& diffxi,
										 const glm::vec2& local,
									     const glm::vec2& scale,
									     float lambdaZ)
{
	glm::vec2 g = local / scale;
	float gauss1 = -0.5f * (g.x * g.x + g.y * g.y);
	float gauss2 = lambdaZ * glm::dot(diffxi, diffxi);
	return exp(gauss1 - gauss2);
}

inline __device__ glm::vec3 RayPlaneIntersect(const glm::vec3 origin, const glm::vec3& direction, const glm::vec3& planePoint, const glm::vec3& planeNormal, bool& found)
{
	float denom = glm::dot(planeNormal, direction);
	found &= glm::abs(denom) > 1e-4f;
	return origin + glm::dot(planeNormal, planePoint - origin) / denom * direction;
}

inline __device__ float ComputeSdf(const glm::vec3& rayOrigin,
								   const glm::vec3& rayDirection,
								   const Nimbus::PrimitivePoint* surfacePoints,
								   const Nimbus::IEPrimitiveInfo& primitiveInfo,
								   const glm::vec3& x,
								   const glm::vec2& scale,
								   float lambdaDistance,
								   glm::vec3& resultNormal,
								   uint32_t& primitivePointID)
{
	float closestPointDistSq = FLT_MAX;
	float denominator = 0.0f;
	glm::vec3 p = glm::vec3(0.0f);
	resultNormal = glm::vec3(0.0f);

	for (uint32_t localPointIndex = 0; localPointIndex < primitiveInfo.pointIndexInfo.count; ++localPointIndex)
	{
		const Nimbus::PrimitivePoint& primitive = surfacePoints[primitiveInfo.pointIndexInfo.first + localPointIndex];
		bool intersect = true;
		glm::vec3 i = RayPlaneIntersect(rayOrigin, rayDirection, primitive.position, primitive.normal, intersect);
		glm::vec3 diffiu = i - primitive.position;
		glm::vec3 diffxi = x - i;

		glm::vec3 right{}, up{};
		Nimbus::Utils::GetOrientationVectors(primitive.normal, right, up);
		glm::vec2 local = glm::vec2(glm::dot(right, diffiu), glm::dot(up, diffiu));
		
		if (!intersect || (abs(local.x) > scale.x || abs(local.y) > scale.y))
			continue;
		
		float gaussian = EvaluateGaussian(primitive, diffxi, local, scale, lambdaDistance);
	
		denominator += gaussian;
		p += gaussian * i;
		resultNormal += gaussian * Nimbus::Utils::FixNormal(rayDirection, primitive.normal);
		float distanceSq = glm::dot(diffxi, diffxi);
		primitivePointID = distanceSq < closestPointDistSq ? primitiveInfo.pointIndexInfo.first + localPointIndex : primitivePointID;
		closestPointDistSq = distanceSq < closestPointDistSq ? distanceSq : closestPointDistSq;
	}
	p /= denominator;
	resultNormal /= denominator;

	return glm::dot(x - p, resultNormal);
}

inline __device__ bool Sdf(const Nimbus::PrimitivePoint* surfacePoints,
							const Nimbus::IEPrimitiveInfo& primitiveInfo,
							const glm::vec3& initialPos,
							const glm::vec3& rayOrigin,
							const glm::vec3& rayDir,
							float sampleDistance,
							float pointRadius,
							float sdfThreshold,
							float lambdaDistance,
							glm::vec3& resultPos,
							glm::vec3& resultNormal,
							float& resultSdf,
							uint32_t& resultPrimitivePointIndex)
{
	glm::vec3 normal = {};
	uint32_t ppID = Nimbus::Constants::InvalidPointIndex;
	glm::vec2 scale = glm::vec2(pointRadius, pointRadius);
	float sdf = ComputeSdf(rayOrigin, rayDir, surfacePoints, primitiveInfo, initialPos, scale, lambdaDistance, normal, ppID);
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
		float dist = glm::isnan(sdf) ? pointRadius : glm::abs(sdf);
		travelDist += dist;
		glm::vec3 cp = p + rayDir * dist;
		glm::vec3 cNormal = glm::vec3(0.0f);
		uint32_t cPpID = Nimbus::Constants::InvalidPointIndex;
		float cSdf = ComputeSdf(rayOrigin, rayDir, surfacePoints, primitiveInfo, cp, scale, lambdaDistance, cNormal, cPpID);
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

inline __device__ bool IntersectWithImplicitSurface(const glm::vec3& rayOrigin,
													const glm::vec3& rayDirection,
													const glm::vec3& rayDestination,
													const Nimbus::IEPrimitiveInfo& primitiveInfo,
													const Nimbus::RayTracingParams& rtParams,
													float& resultDistance,
													glm::vec3& resultNormal,
													uint32_t& resultPrimitivePointIndex)
{
	glm::vec3 closestPoint = Nimbus::Utils::ProjectPointToRay(rayOrigin, rayDirection, rayDestination);
	float t0 = glm::length(rayOrigin - (closestPoint - rayDirection * (rtParams.env.pc.sampleDistance * 0.5f)));

	glm::vec3 resPos{};
	glm::vec3 resNorm{};
	float sdf{};
	bool found = Sdf(rtParams.env.pc.primitivePoints,
					 primitiveInfo,
					 rayOrigin + rayDirection * t0,
					 rayOrigin,
					 rayDirection,
					 rtParams.env.pc.sampleDistance,
					 rtParams.env.pc.pointRadius,
					 rtParams.env.pc.sdfThreshold,
					 rtParams.env.pc.lambdaDistance,
					 resPos,
					 resNorm,
					 sdf,
					 resultPrimitivePointIndex);


	glm::vec3 normal = glm::normalize(resNorm);
	glm::vec3 planePoint = resPos + glm::abs(sdf) * -normal;
	resultDistance = glm::length(RayPlaneIntersect(rayOrigin, rayDirection, planePoint, normal, found) - rayOrigin);
	resultNormal = normal;
	return found;
}