#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <Types.hpp>
#include <Utils.hpp>
#include "Ray.cuh"

class PathRefiner
{
public:
	__device__ PathRefiner(const VCT::SceneData& sceneData, const VCT::ConeTracingData& coneTracingData, const VCT::TraceData& traceData);
	__device__ bool Refine(const VCT::RefineParams& params, VCT::TraceData& result);

private:
	__device__ float InitializeRefineData(const VCT::TraceData& traceData);
	__device__ uint32_t NumPoints() const;
	__device__ float GetPathLength() const;
	__device__ float NormalizePath();
	__device__ float f(const glm::vec3& point, uint32_t iaIndex) const;
	__device__ glm::vec2 fGradient(uint32_t iaIndex) const;
	__device__ float LineSearch(uint32_t iaIndex, const VCT::RefineParams& params) const;
	__device__ bool ValidatePath(VCT::TraceData& result);
	__device__ bool ValidateInteractionDirection(const glm::vec3& direction, const glm::vec3& normal, VCT::InteractionType iaType) const;

private:
	struct RefineData
	{
		glm::vec3 position;
		glm::vec3 normalizedPosition;
		glm::vec3 u;
		glm::vec3 v;
		glm::vec3 normal;
		uint32_t primitivePointID;
		glm::vec2 gradient;
		uint32_t ieID;
		uint32_t parentID;
		VCT::InteractionType iaType;
	};

private:
	const VCT::SceneData& m_SceneData;
	const VCT::ConeTracingData& m_ConeTracingData;
	RefineData m_RefineData[VCT::Constants::MaximumNumberOfInteractions + 2];
	uint32_t m_TxID;
	uint32_t m_RxID;
	uint32_t m_NumInteractions;
	float m_PathLength;
};

inline __device__ PathRefiner::PathRefiner(const VCT::SceneData& sceneData, const VCT::ConeTracingData& coneTracingData, const VCT::TraceData& traceData)
	: m_SceneData(sceneData)
	, m_ConeTracingData(coneTracingData)
	, m_RefineData()
	, m_TxID(traceData.transmitterID)
	, m_RxID(traceData.receiverID)
	, m_NumInteractions(traceData.numInteractions)
	, m_PathLength(InitializeRefineData(traceData))
{
}

inline __device__ bool PathRefiner::Refine(const VCT::RefineParams& params, VCT::TraceData& result)
{
	bool converged = false;
	for (uint32_t it = 0; it < params.numIterations; ++it)
	{
		float normSq = 0.0f;
		bool fail = false;
		for (uint32_t i = 1; i <= m_NumInteractions; ++i)
		{
			RefineData& rd = m_RefineData[i];
			rd.gradient = fGradient(i);
			normSq += dot(rd.gradient, rd.gradient);
			rd.normalizedPosition = rd.normalizedPosition + (-rd.gradient.x * rd.u + -rd.gradient.y * rd.v) * LineSearch(i, params);
			float nLen = glm::length(rd.normalizedPosition - m_RefineData[i - 1].normalizedPosition);
			glm::vec3 nDir = (rd.normalizedPosition - m_RefineData[i - 1].normalizedPosition) / nLen;
			glm::vec3 rtPos = m_RefineData[i - 1].position + nDir * (nLen * m_PathLength);

			if (rd.iaType == VCT::InteractionType::Reflection)
			{
				Ray ray(m_RefineData[i - 1].position, rtPos);
				
				//if (fail = !ray.Trace(m_SceneData.refineRtParams.asHandle, m_SceneData.refineRtParams.traceDistanceBias, m_SceneData.refineRtParams.traceDistanceBias))
				if (fail = !ray.Trace(m_SceneData.refineRtParams.asHandle, ray.AbsoluteDistance() - m_SceneData.refineRtParams.traceDistanceBias, m_SceneData.refineRtParams.traceDistanceBias))
					break;

				glm::vec3 newPos = ray.GetOrigin() + ray.GetDirection() * ray.GetPayload().GetDistance();
				glm::vec3 newNorm = VCT::Utils::FixNormal(ray.GetDirection(), ray.GetPayload().GetNormal());
				float planeSdf = glm::abs(glm::dot(newPos - rd.position, rd.normal));
				if (glm::dot(rd.normal, newNorm) < params.angleThreshold || planeSdf > params.distanceThreshold)
				{
					//if change between iterations is too big, update normal vector and position
					rd.position = ray.GetOrigin() + ray.GetDirection() * ray.GetPayload().GetDistance();
					rd.normal = VCT::Utils::FixNormal(ray.GetDirection(), ray.GetPayload().GetNormal());
					rd.primitivePointID = ray.GetPayload().GetPrimitivePointID();
					rd.ieID = ray.GetPayload().hitIeID;
					VCT::Utils::GetOrientationVectors(rd.normal, rd.u, rd.v);
					m_PathLength = NormalizePath();
				}
				else
				{
					//Else we move on the same plane to counter noisiness
					glm::vec3 origin = m_RefineData[i - 1].position;
					glm::vec3 dir = ray.GetDirection();
					float denom = glm::dot(rd.normal, dir);
					rd.position = origin + glm::dot(rd.normal, rd.position - origin) / denom * dir;
					rd.ieID = ray.GetPayload().hitIeID;
					rd.primitivePointID = ray.GetPayload().GetPrimitivePointID();
					m_PathLength = NormalizePath();
				}
			}

			else if (rd.iaType == VCT::InteractionType::Diffraction)
			{
				glm::vec3 origin = m_RefineData[i - 1].position;
				glm::vec3 dir = glm::normalize(rd.normalizedPosition - m_RefineData[i - 1].normalizedPosition);
				glm::vec3 cr = glm::cross(rd.u, dir);
				glm::vec3 pos = origin + dir * glm::abs(glm::dot(glm::cross(rd.position - origin, rd.u), cr) / glm::dot(cr, cr));

				const VCT::DiffractionEdge& edge = m_ConeTracingData.diffractionEdges[rd.parentID];
				if (fail = !VCT::Utils::IsPointOnLine(edge.startPoint, edge.endPoint, rtPos))
					break;
				rd.position = rtPos;
				m_PathLength = NormalizePath();
			}
		}

		if (fail)
			break;
		
		converged = !fail && normSq < params.delta;
	}
	return converged && ValidatePath(result);
}

inline __device__ float PathRefiner::InitializeRefineData(const VCT::TraceData& traceData)
{
	m_RefineData[0].position = m_SceneData.transmitters[traceData.transmitterID].position;
	m_RefineData[traceData.numInteractions + 1].position = m_SceneData.receivers[traceData.transmitterID].position;

	for (uint32_t iaIndex = 0; iaIndex < traceData.numInteractions; ++iaIndex)
	{
		RefineData& rd = m_RefineData[iaIndex + 1];
		const VCT::Interaction& ia = traceData.interactions[iaIndex];
		rd.position = ia.position;
		rd.normal = ia.normal;
		rd.ieID = ia.ieID;
		rd.iaType = ia.type;
		rd.gradient = glm::vec2(0.0f);

		if (traceData.interactions[iaIndex].type == VCT::InteractionType::Diffraction)
		{
			rd.parentID = m_ConeTracingData.diffractionEdgeSegments[m_SceneData.intersectableEntities[traceData.interactions[iaIndex].ieID].edgeSegmentID].parentID;
			rd.u = m_ConeTracingData.diffractionEdges[rd.parentID].forward;
			rd.v = glm::vec3(0.0f);
		}
		else
		{
			Ray ray(m_RefineData[iaIndex].position, rd.position);
			if (ray.Trace(m_SceneData.refineRtParams.asHandle, 0.0f, m_SceneData.refineRtParams.traceDistanceBias))
			{
				rd.position = ray.GetOrigin() + ray.GetDirection() * ray.GetPayload().GetDistance();
				rd.normal = VCT::Utils::FixNormal(ray.GetDirection(), ray.GetPayload().GetNormal());
				rd.ieID = ray.GetPayload().hitIeID;
				rd.primitivePointID = ray.GetPayload().GetPrimitivePointID();
			}
			
			VCT::Utils::GetOrientationVectors(rd.normal, rd.u, rd.v);
		}
	}
	return NormalizePath();
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

inline __device__ float PathRefiner::f(const glm::vec3& point, uint32_t iaIndex) const
{
	return glm::length(point - m_RefineData[iaIndex - 1].normalizedPosition) + glm::length(point - m_RefineData[iaIndex + 1].normalizedPosition);
}

inline __device__ glm::vec2 PathRefiner::fGradient(uint32_t iaIndex) const
{
	glm::vec3 p = glm::normalize(m_RefineData[iaIndex].normalizedPosition - m_RefineData[iaIndex - 1].normalizedPosition) + glm::normalize(m_RefineData[iaIndex].normalizedPosition - m_RefineData[iaIndex + 1].normalizedPosition);
	return glm::vec2(glm::dot(p, m_RefineData[iaIndex].u), glm::dot(p, m_RefineData[iaIndex].v));
}

inline __device__ float PathRefiner::LineSearch(uint32_t iaIndex, const VCT::RefineParams& params) const
{
	float stepSize = 1.f;
	glm::vec3 gradientModifier = m_RefineData[iaIndex].gradient.x * m_RefineData[iaIndex].u + m_RefineData[iaIndex].gradient.y * m_RefineData[iaIndex].v;
	float gradientNormSq = glm::dot(m_RefineData[iaIndex].gradient, m_RefineData[iaIndex].gradient);
	float fx = f(m_RefineData[iaIndex].normalizedPosition, iaIndex);

	while (f(m_RefineData[iaIndex].normalizedPosition - stepSize * gradientModifier, iaIndex) > fx - params.alpha * stepSize * gradientNormSq)
		stepSize *= params.beta;

	return stepSize;
}

inline __device__ bool PathRefiner::ValidatePath(VCT::TraceData& result)
{
	bool validPath = true;
	for (uint32_t i = 0; i < m_NumInteractions; ++i)
	{
		result.numInteractions = m_NumInteractions;
		result.transmitterID = m_TxID;
		result.receiverID = m_RxID;
		result.interactions[i].position = m_RefineData[i + 1].position;
		result.interactions[i].normal = m_RefineData[i + 1].normal;
		result.interactions[i].curvature = m_RefineData[i + 1].primitivePointID;
		result.interactions[i].type = m_RefineData[i + 1].iaType;

		if (result.interactions[i].type == VCT::InteractionType::Diffraction)
		{
			const VCT::DiffractionEdge& edge = m_ConeTracingData.diffractionEdges[m_RefineData[i + 1].parentID];
			validPath &= VCT::Utils::IsPointOnLine(edge.startPoint, edge.endPoint, result.interactions[i].position);
			result.interactions[i].label = m_RefineData[i + 1].parentID;
		}
		else
		{
			result.interactions[i].label = m_SceneData.refineRtParams.primitivePoints[m_RefineData[i + 1].primitivePointID].label;
		}
		if (i < m_NumInteractions - 1)
		{
			Ray ray(m_RefineData[i + 1].position, m_RefineData[i + 2].position);
			uint32_t dstIeID = m_RefineData[i + 2].ieID;
			validPath &= ray.Trace(m_SceneData.refineRtParams, dstIeID, m_SceneData.intersectableEntities[dstIeID].type);
		}
		validPath &= ValidateInteractionDirection(glm::normalize(m_RefineData[i + 2].position - m_RefineData[i + 1].position), m_RefineData[i + 1].normal, m_RefineData[i + 1].iaType);
	}
	Ray txRay(m_RefineData[0].position, m_RefineData[1].position);
	validPath &= txRay.Trace(m_SceneData.refineRtParams, m_RefineData[1].ieID, m_SceneData.intersectableEntities[m_RefineData[1].ieID].type);

	Ray rxRay = Ray(m_RefineData[m_NumInteractions].position, m_RefineData[m_NumInteractions + 1].position);
	validPath &= !rxRay.Trace(m_SceneData.refineRtParams.asHandle, 0.0f, -m_SceneData.refineRtParams.traceDistanceBias);

	return validPath;
}

inline __device__ bool PathRefiner::ValidateInteractionDirection(const glm::vec3& direction, const glm::vec3& normal, VCT::InteractionType iaType) const
{
	if (VCT::IsInteractionType<VCT::InteractionType::Reflection>(iaType))
		return glm::dot(direction, normal) >= 0.0f;

	if (VCT::IsInteractionType<VCT::InteractionType::Diffraction>(iaType))
		return true;
}