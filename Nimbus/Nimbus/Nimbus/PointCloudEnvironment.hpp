#pragma once
#include "CudaUtils.hpp"
#include "Environment.hpp"
#include <memory>

namespace Nimbus
{
	class PointCloudEnvironment : public Environment
	{
	public:
		PointCloudEnvironment();
		bool Init(const PointData* points,
				  size_t numPoints,
				  const EdgeData* edges,
				  size_t numEdges,
				  float voxelSize,
				  float pointRadius,
				  float sdfThreshold,
				  float lambdaDistance);

		float GetVoxelSize() const override { return m_VoxelSize; }
		float GetPointRadius() const { return m_PointRadius; }
		float GetSdfThreshold() const { return m_SdfThreshold; }
		float GetLambdaDistance() const { return m_LambdaDistance; }
		Type GetType() const override { return Type::PointCloud; }
		EnvironmentData GetGpuEnvironmentData()override;
		const Aabb& GetAabb() const override { return m_Aabb; }
		uint32_t GetRtPointCount() const override { return m_IeCount; }

		void ComputeVisibility(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void DetermineLosPaths(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void Transmit(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void Propagate(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void RefineSpecular(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void RefineScatterer(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void RefineDiffraction(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void ComputeRISPaths(const DeviceBuffer& params, const glm::uvec3& dims) const override;

	private:
		std::vector<PointNode> LoadPoints(const PointData* points, size_t numPoints);
		bool ComputeVoxelWorld(float voxelSize);
		std::vector<glm::uvec2> LinkPointNodes(std::vector<PointNode>& pointNodes);
		bool GenerateRayTracingData(const std::vector<PointNode>& pointNodes, const std::vector<glm::uvec2>& voxelNodeIndices);

	private:
		float m_VoxelSize;
		float m_PointRadius;
		float m_SdfThreshold;
		float m_LambdaDistance;
		Aabb m_Aabb;
		VoxelWorldInfo m_VoxelWorldInfo;
		uint32_t m_IeCount;
		uint32_t m_PointCount;
		DeviceBuffer m_PrimitiveBuffer;
		DeviceBuffer m_RtPointBuffer;
		DeviceBuffer m_PrimitiveInfoBuffer;
		DeviceBuffer m_PrimitivePointBuffer;
	};
}