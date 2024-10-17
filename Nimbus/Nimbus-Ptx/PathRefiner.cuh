#pragma once
#include "Ray.cuh"
#include "Nimbus/Types.hpp"
#include "Nimbus/Utils.hpp"

struct RefineData
{
	glm::vec3 position;
	glm::vec3 normalizedPosition;
	glm::vec3 u;
	glm::vec3 v;
	glm::vec3 normal;
	uint32_t label;
	uint32_t material;
	glm::vec2 gradient;
};

class PathRefiner
{
public:
	__device__ PathRefiner(const Nimbus::PathInfoST& pathInfo,
						   const glm::vec3* interactions,
						   const glm::vec3* normals,
						   const glm::vec3& tx,
						   const glm::vec3& rx,
						   uint32_t rxID,
						   const Nimbus::RayTracingParams& params);

	inline __device__ bool Refine(const Nimbus::RefineParams& params, Nimbus::PathInfo& result);
	inline __device__ const RefineData* GetRefineData() const;

private:
	__device__ float InitializeRefineData(const Nimbus::PathInfoST& pathInfo, const glm::vec3* interactions, const glm::vec3* normals, const glm::vec3& tx, const glm::vec3& rx);
	__device__ bool RefineFast(const Nimbus::RefineParams& params);
	__device__ bool IsRefineAccurate(const Nimbus::RefineParams& params);
	__device__ bool ValidateReflectionVisibility(uint32_t ia);
	__device__ bool WriteResult(Nimbus::PathInfo& result);
	__device__ float f(const glm::vec3& point, uint32_t iaIndex) const;
	__device__ glm::vec2 fGradient(uint32_t iaIndex) const;
	__device__ float LineSearch(uint32_t iaIndex, const Nimbus::RefineParams& params) const;
	__device__ uint32_t NumPoints() const;
	__device__ float GetPathLength() const;
	__device__ float NormalizePath();

private:
	RefineData m_RefineData[Nimbus::Constants::MaximumNumberOfInteractions + 2];
	Nimbus::RayTracingParams m_Params;
	uint32_t m_TxID;
	uint32_t m_RxID;
	uint32_t m_NumInteractions;
	float m_PathLength;
};

inline __device__ PathRefiner::PathRefiner(const Nimbus::PathInfoST& pathInfo,
										   const glm::vec3* interactions,
										   const glm::vec3* normals,
										   const glm::vec3& tx,
										   const glm::vec3& rx,
										   uint32_t rxID,
										   const Nimbus::RayTracingParams& params)
	: m_Params(params)
	, m_TxID(pathInfo.txID)
	, m_RxID(rxID)
	, m_NumInteractions(pathInfo.numInteractions)
	, m_PathLength(InitializeRefineData(pathInfo, interactions, normals, tx, rx))
{

}

inline __device__ bool PathRefiner::Refine(const Nimbus::RefineParams& params, Nimbus::PathInfo& result)
{
	for (uint32_t i = 0; i < params.maxCorrectionIterations; ++i)
	{
		if (!RefineFast(params))
		{
			return false;
		}
		if (IsRefineAccurate(params))
		{
			return WriteResult(result);
		}
	}
	return false;
}

inline __device__ const RefineData* PathRefiner::GetRefineData() const
{
	return m_RefineData;
}

inline __device__ float PathRefiner::InitializeRefineData(const Nimbus::PathInfoST& pathInfo,
														  const glm::vec3* interactions,
														  const glm::vec3* normals,
														  const glm::vec3& tx,
														  const glm::vec3& rx)
{
	m_RefineData[0].position = tx;
	m_RefineData[pathInfo.numInteractions + 1].position = rx;

	for (uint32_t iaIndex = 0; iaIndex < pathInfo.numInteractions; ++iaIndex)
	{
		RefineData& rd = m_RefineData[iaIndex + 1];
		rd.position = interactions[iaIndex];
		rd.normal = normals[iaIndex];
		rd.gradient = glm::vec2(0.0f);

		Ray ray(m_RefineData[iaIndex].position, rd.position);
		if (ray.Trace(m_Params.env.asHandle, m_Params.rayBias, m_Params.rayBias))
		{
			rd.position = ray.GetOrigin() + ray.GetDirection() * ray.GetPayload().t;
			rd.normal = Nimbus::Utils::FixNormal(ray.GetDirection(), ray.GetPayload().normal);
			rd.label = ray.GetPayload().label;
			rd.material = ray.GetPayload().material;
		}
		Nimbus::Utils::GetOrientationVectors(rd.normal, rd.u, rd.v);
	}
	return NormalizePath();
}

inline __device__ bool PathRefiner::RefineFast(const Nimbus::RefineParams& params)
{
	for (uint32_t it = 0; it < params.maxNumIterations; ++it)
	{
		float normSq = 0.0f;
		for (uint32_t i = 1; i <= m_NumInteractions; ++i)
		{
			RefineData& rd = m_RefineData[i];
			rd.gradient = fGradient(i);
			normSq += dot(rd.gradient, rd.gradient);
			rd.normalizedPosition = rd.normalizedPosition + (-rd.gradient.x * rd.u + -rd.gradient.y * rd.v) * LineSearch(i, params);
		}

		if (normSq < params.delta)
			return true;
	}
	return false;
}

inline __device__ bool PathRefiner::IsRefineAccurate(const Nimbus::RefineParams& params)
{
	bool pathAccurate = true;
	for (uint32_t i = 1; i <= m_NumInteractions; ++i)
	{
		RefineData& rd = m_RefineData[i];
		Ray ray(m_RefineData[i - 1].position, m_RefineData[i - 1].position + glm::normalize(rd.normalizedPosition - m_RefineData[i - 1].normalizedPosition) * m_Params.rayMaxLength);
		if (ray.Trace(m_Params.env.asHandle, m_Params.rayBias, m_Params.rayBias))
		{
			glm::vec3 newPos = ray.GetOrigin() + ray.GetDirection() * ray.GetPayload().t;
			glm::vec3 newNorm = Nimbus::Utils::FixNormal(ray.GetDirection(), ray.GetPayload().normal);
			float planeSdf = glm::abs(glm::dot(newPos - rd.position, rd.normal));
			bool notAccurateNode = glm::dot(rd.normal, newNorm) < params.angleThreshold || planeSdf > params.distanceThreshold;
			rd.position = ray.GetOrigin() + ray.GetDirection() * ray.GetPayload().t;
			rd.normal = Nimbus::Utils::FixNormal(ray.GetDirection(), ray.GetPayload().normal);
			rd.label = ray.GetPayload().label;
			rd.material = ray.GetPayload().material;
			Nimbus::Utils::GetOrientationVectors(rd.normal, rd.u, rd.v);
			pathAccurate &= !notAccurateNode;
		}
	}
	m_PathLength = NormalizePath();
	return pathAccurate;
}

inline __device__ bool PathRefiner::ValidateReflectionVisibility(uint32_t ia)
{
	auto& src = m_RefineData[ia + 2];
	auto& dst = m_RefineData[ia + 1];
	Ray ray(src.position, dst.position);
	bool res = ray.Trace(m_Params.env.asHandle, m_Params.rayBias, m_Params.rayBias);
	
	return res && glm::dot(ray.GetPayload().normal, dst.normal) > 0.99f
		&& dst.label == ray.GetPayload().label;
}

inline __device__ bool PathRefiner::WriteResult(Nimbus::PathInfo& result)
{
	bool pathValid = true;
	result.txID = m_TxID;
	result.rxID = m_RxID;
	result.numInteractions = m_NumInteractions;
	result.timeDelay = glm::length(m_RefineData[0].position - m_RefineData[1].position) * Nimbus::Constants::InvLightSpeedInVacuum;
	for (uint32_t ia = 0; ia < m_NumInteractions; ++ia)
	{
		result.timeDelay += glm::length(m_RefineData[ia + 1].position - m_RefineData[ia + 2].position) * Nimbus::Constants::InvLightSpeedInVacuum;
		glm::vec3 incident = glm::normalize(m_RefineData[ia + 1].position - m_RefineData[ia + 0].position);
		glm::vec3 reflected = glm::normalize(m_RefineData[ia + 2].position - m_RefineData[ia + 1].position);
		pathValid &= glm::dot(glm::reflect(incident, m_RefineData[ia + 1].normal), reflected) > 0.99f;
		pathValid &= ValidateReflectionVisibility(ia);
	}
	Ray ray(m_RefineData[m_NumInteractions + 1].position, m_RefineData[m_NumInteractions].position);
	pathValid &= ray.Trace(m_Params.env.asHandle, 0.0f, m_Params.rayBias);
	pathValid &= glm::dot(ray.GetPayload().normal, m_RefineData[m_NumInteractions].normal) > 0.99f;

	return pathValid;
}

inline __device__ float PathRefiner::f(const glm::vec3& point, uint32_t iaIndex) const
{
	return glm::length(point - m_RefineData[iaIndex - 1].normalizedPosition) + glm::length(point - m_RefineData[iaIndex + 1].normalizedPosition);
}

inline __device__ glm::vec2 PathRefiner::fGradient(uint32_t iaIndex) const
{
	glm::vec3 p = glm::normalize(m_RefineData[iaIndex].normalizedPosition - m_RefineData[iaIndex - 1].normalizedPosition) 
				+ glm::normalize(m_RefineData[iaIndex].normalizedPosition - m_RefineData[iaIndex + 1].normalizedPosition);
	return glm::vec2(glm::dot(p, m_RefineData[iaIndex].u), glm::dot(p, m_RefineData[iaIndex].v));
}

inline __device__ float PathRefiner::LineSearch(uint32_t iaIndex, const Nimbus::RefineParams& params) const
{
	float stepSize = 1.f;
	glm::vec3 gradientModifier = m_RefineData[iaIndex].gradient.x * m_RefineData[iaIndex].u + m_RefineData[iaIndex].gradient.y * m_RefineData[iaIndex].v;
	float gradientNormSq = glm::dot(m_RefineData[iaIndex].gradient, m_RefineData[iaIndex].gradient);
	float fx = f(m_RefineData[iaIndex].normalizedPosition, iaIndex);

	while (f(m_RefineData[iaIndex].normalizedPosition - stepSize * gradientModifier, iaIndex) > fx - params.alpha * stepSize * gradientNormSq)
		stepSize *= params.beta;

	return stepSize;
}

inline __device__ uint32_t PathRefiner::NumPoints() const
{
	return m_NumInteractions + 2;
}

inline __device__ float PathRefiner::GetPathLength() const
{
	float pathLength = 0.0f;

	for (uint32_t i = 1; i < NumPoints(); ++i)
		pathLength += glm::length(m_RefineData[i].position - m_RefineData[i - 1].position);

	return pathLength;
}

inline __device__ float PathRefiner::NormalizePath()
{
	float len = GetPathLength();
	float invLen = 1.0f / len;
	m_RefineData[0].normalizedPosition = glm::vec3(0.0f);
	for (uint32_t i = 1; i < NumPoints(); ++i)
		m_RefineData[i].normalizedPosition = m_RefineData[i - 1].normalizedPosition + (m_RefineData[i].position - m_RefineData[i - 1].position) * invLen;

	return len;
}
