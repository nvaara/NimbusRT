#pragma once
#include "Types.hpp"
#include "Utils.hpp"

namespace VCT
{
	class VoxelTraverser
	{
	public:
		__device__ VoxelTraverser(const glm::vec3& voxelSpacePosition, const glm::vec3& rayDirection);
		__device__ VoxelTraverser& Step(uint32_t additionalSteps);
		__device__ const glm::vec3& GetTraverseVoxel() const;
		__device__ const glm::vec3& GetCurrentVoxel() const;
		__device__ glm::vec3 GetTextureVoxel() const;
		__device__ glm::vec3 GetRayDirection() const;

	private:
		__device__ float RayMarch(uint32_t additionalSteps);

	private:
		static constexpr float ErrorBias = 1e-2f;
		static constexpr float ZeroDivisionBias = 1e-16f;
		glm::vec3 m_Voxel;
		glm::vec3 m_RayDirection;
		glm::vec3 m_LocalStepDirection;
		glm::vec3 m_StepsPerUnit;
		glm::vec3 m_CurrentVoxel;
	};

	inline __device__ VoxelTraverser::VoxelTraverser(const glm::vec3& voxelSpacePosition, const glm::vec3& rayDirection)
		: m_Voxel(voxelSpacePosition)
		, m_RayDirection(rayDirection)
		, m_LocalStepDirection(glm::vec3(float(rayDirection.x >= 0), float(rayDirection.y >= 0), float(rayDirection.z >= 0)))
		, m_StepsPerUnit(1.0f / (glm::max)(glm::abs(rayDirection), ZeroDivisionBias))
		, m_CurrentVoxel(floor(m_Voxel))
	{
	}

	inline __device__ VoxelTraverser& VoxelTraverser::Step(uint32_t additionalSteps)
	{
		float stepSize = RayMarch(additionalSteps);
		m_Voxel += m_RayDirection * (stepSize + ErrorBias);
		m_CurrentVoxel = floor(m_Voxel);
		return *this;
	}

	inline __device__ const glm::vec3& VoxelTraverser::GetTraverseVoxel() const
	{
		return m_Voxel;
	}

	inline __device__ const glm::vec3& VoxelTraverser::GetCurrentVoxel() const
	{
		return m_CurrentVoxel;
	}

	inline __device__ glm::vec3 VoxelTraverser::GetTextureVoxel() const
	{
		return m_CurrentVoxel + 0.5f;
	}

	inline __device__ glm::vec3 VoxelTraverser::GetRayDirection() const
	{
		return m_RayDirection;
	}

	inline __device__ float VoxelTraverser::RayMarch(uint32_t additionalSteps)
	{
		glm::vec3 localPosition = m_Voxel - m_CurrentVoxel;
		glm::vec3 distToNextVoxel = glm::abs(m_LocalStepDirection - localPosition);
		glm::vec3 stepsToNextVoxel = distToNextVoxel * m_StepsPerUnit;
		glm::vec3 remainingVoxels = m_StepsPerUnit * static_cast<float>(additionalSteps);
		glm::vec3 totalSteps = stepsToNextVoxel + remainingVoxels;

		glm::vec3 multiplier = glm::vec3(0.0f);
		multiplier.x = static_cast<float>((totalSteps.x <= totalSteps.y) & (totalSteps.x <= totalSteps.z));
		multiplier.y = static_cast<float>((totalSteps.y < totalSteps.x) & (totalSteps.y <= totalSteps.z));
		multiplier.z = static_cast<float>((totalSteps.z < totalSteps.x) & (totalSteps.z < totalSteps.y));
		return glm::dot(totalSteps, multiplier);
	}
}