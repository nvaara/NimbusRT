#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "Types.hpp"
#include <optix.h>

class Ray
{
public:
	static constexpr float RayBias = 0.01f;

	__device__ Ray(const glm::vec3& rayOrigin, const glm::vec3& rayDestination);
	__device__ bool Trace(const VCT::RayTracingParams& rtParams, uint32_t ieID, VCT::IEType ieType);
	__device__ bool Trace(OptixTraversableHandle asHandle, float minLenBias, float maxLenBias);

	__device__ const glm::vec3& GetOrigin() const { return m_Origin; }
	__device__ const glm::vec3& GetDirection() const { return m_Direction; }
	__device__ float AbsoluteDistance() const { return m_Distance; }

	struct Payload
	{
		inline __device__ Payload() : hitIeID(VCT::Constants::InvalidPointIndex), ieType(VCT::Constants::InvalidPointIndex), distance(0), normal(0) {}
		inline __device__ uint32_t GetHitIeID() const { return hitIeID; }
		inline __device__ float GetDistance() const { return __uint_as_float(distance); }
		inline __device__ glm::vec3 GetNormal() const { return glm::vec3(__uint_as_float(normal.x), __uint_as_float(normal.y), __uint_as_float(normal.z)); }
		inline __device__ VCT::IEType GetSurfaceType() const { return static_cast<VCT::IEType>(ieType); }
		inline __device__ uint32_t GetPrimitivePointID() const { return primitivePointID; }
		inline __device__ uint32_t GetPrimitiveID() const { return primitiveID; }

		uint32_t hitIeID;
		uint32_t ieType;
		uint32_t distance;
		glm::uvec3 normal;
		union
		{
			uint32_t primitivePointID;
			uint32_t primitiveID;
		};
	};

	__device__ const Payload& GetPayload() const { return m_Payload; }

private:
	static __device__ float GetSurfaceDistanceBias(float bias, VCT::IEType ieType);
	static constexpr float RayTime = 0.0f;
	static constexpr uint32_t VisMask = 1;
	static constexpr uint32_t TraceFlags =  OPTIX_RAY_FLAG_DISABLE_ANYHIT;

private:
	glm::vec3 m_Origin;
	glm::vec3 m_Direction;
	float m_Distance;
	Payload m_Payload;
};

inline __device__ Ray::Ray(const glm::vec3& rayOrigin, const glm::vec3& rayDestination)
	: m_Origin(rayOrigin)
	, m_Payload()
{
	glm::vec3 v = rayDestination - rayOrigin;
	m_Distance = glm::length(v);
	m_Direction = v * (1.0f / m_Distance);
}

__device__ bool Ray::Trace(const VCT::RayTracingParams& rtParams, uint32_t ieID, VCT::IEType ieType)
{
	m_Payload.hitIeID = ieID;
	m_Payload.ieType = static_cast<uint32_t>(ieType);
	float traceDistance = m_Distance + GetSurfaceDistanceBias(rtParams.traceDistanceBias, ieType);
	
	optixTrace(rtParams.asHandle,
			   reinterpret_cast<const float3&>(m_Origin),
			   reinterpret_cast<const float3&>(m_Direction),
			   RayBias, traceDistance, RayTime, VisMask, TraceFlags, 0u, 0u, 0u,
			   m_Payload.hitIeID, m_Payload.ieType, m_Payload.distance,
			   m_Payload.normal.x, m_Payload.normal.y, m_Payload.normal.z, m_Payload.primitivePointID);
	
	return m_Payload.hitIeID == ieID;
}

__device__ bool Ray::Trace(OptixTraversableHandle asHandle, float minLenBias, float maxLenBias)
{
	m_Payload.ieType = static_cast<uint32_t>(VCT::IEType::Surface);
	optixTrace(asHandle,
		reinterpret_cast<const float3&>(m_Origin),
		reinterpret_cast<const float3&>(m_Direction),
		RayBias + minLenBias, m_Distance + RayBias + maxLenBias, RayTime, VisMask, TraceFlags, 0u, 0u, 0u,
		m_Payload.hitIeID, m_Payload.ieType, m_Payload.distance,
		m_Payload.normal.x, m_Payload.normal.y, m_Payload.normal.z, m_Payload.primitivePointID);

	return m_Payload.hitIeID != VCT::Constants::InvalidPointIndex;
}

inline __device__ float Ray::GetSurfaceDistanceBias(float bias, VCT::IEType ieType)
{
	return -static_cast<float>(ieType != VCT::IEType::Surface) * bias + static_cast<float>(ieType == VCT::IEType::Surface) * bias;
}