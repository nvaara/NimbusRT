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
		bool Init(const PointData* points, size_t numPoints, const EdgeData* edges, size_t numEdges, float voxelSize, float aabbBias);

		float GetVoxelSize() const override { return m_VoxelSize; }
		Type GetType() const override { return Type::PointCloud; }
		EnvironmentData GetGpuEnvironmentData() const override;
		const Aabb& GetAabb() const override { return m_Aabb; }
		uint32_t GetRtPointCount() const override { return m_IeCount; }
		virtual uint32_t GetManualEdgeCount() const override { return m_EdgeCount; }
		virtual const DeviceBuffer& GetEdgeBuffer() const override { return m_EdgeBuffer; };

		void ComputeVisibility(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void DetermineLosPaths(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void Transmit(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void Propagate(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void RefineSpecular(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void RefineScatterer(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void RefineDiffraction(const DeviceBuffer& params, const glm::uvec3& dims) const override;

	private:
		std::vector<PointNode> LoadPoints(const PointData* points, size_t numPoints);
		bool ComputeVoxelWorld(float voxelSize);
		std::vector<glm::uvec2> LinkPointNodes(std::vector<PointNode>& pointNodes);
		bool GenerateRayTracingData(const std::vector<PointNode>& pointNodes, const std::vector<glm::uvec2>& voxelNodeIndices, float aabbBias);
		bool ProcessEdges(const EdgeData* edges, size_t numEdges);

	private:
		float m_VoxelSize;
		Aabb m_Aabb;
		AccelerationStructure m_AccelerationStructure;
		VoxelWorldInfo m_VoxelWorldInfo;
		uint32_t m_IeCount;
		uint32_t m_PointCount;
		DeviceBuffer m_PrimitiveBuffer;
		DeviceBuffer m_RtPointBuffer;
		DeviceBuffer m_PrimitiveInfoBuffer;
		DeviceBuffer m_PrimitivePointBuffer;
		uint32_t m_EdgeCount;
		DeviceBuffer m_EdgeBuffer;
	};
}