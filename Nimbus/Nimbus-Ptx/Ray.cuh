#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "Nimbus/Types.hpp"
#include <optix.h>

class Ray
{
public:
	__device__ Ray(const glm::vec3& rayOrigin, const glm::vec3& rayDestination);
	__device__ bool Trace(OptixTraversableHandle asHandle, float minLenBias, float maxLenBias, uint32_t additionalFlags = 0);

	__device__ const glm::vec3& GetOrigin() const { return m_Origin; }
	__device__ const glm::vec3& GetDirection() const { return m_Direction; }
	__device__ float AbsoluteDistance() const { return m_Distance; }

	struct Payload
	{
		float t;
		glm::vec3 normal;
		uint32_t rtPointIndex;
		uint32_t label;
		uint32_t material;
	};

	__device__ const Payload& GetPayload() const { return m_Payload; }

private:
	static constexpr float RayTime = 0.0f;
	static constexpr uint32_t VisMask = 0xFF;
	static constexpr uint32_t TraceFlags =  OPTIX_RAY_FLAG_DISABLE_ANYHIT;

private:
	glm::vec3 m_Origin;
	glm::vec3 m_Direction;
	float m_Distance;
	Payload m_Payload;
};

inline __device__ Ray::Ray(const glm::vec3& rayOrigin, const glm::vec3& rayDestination)
	: m_Origin(rayOrigin)
	, m_Payload({})
{
	glm::vec3 v = rayDestination - rayOrigin;
	m_Distance = glm::length(v);
	m_Direction = v * (1.0f / m_Distance);
}

__device__ bool Ray::Trace(OptixTraversableHandle asHandle, float minLenBias, float maxLenBias, uint32_t additionalFlags)
{
	glm::uvec2 ptr = Nimbus::Utils::PackPointer32(&m_Payload);
	optixTrace(asHandle,
		reinterpret_cast<const float3&>(m_Origin),
		reinterpret_cast<const float3&>(m_Direction),
		minLenBias, m_Distance + maxLenBias, RayTime, VisMask, TraceFlags | additionalFlags, 0u, 0u, 0u,
		ptr.x, ptr.y);

	return m_Payload.t > 0.0f;
}