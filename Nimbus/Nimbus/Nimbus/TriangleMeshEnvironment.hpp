#pragma once
#include "Nimbus/Types.hpp"
#include "DeviceBuffer.hpp"
#include "CudaUtils.hpp"
#include "Environment.hpp"

namespace Nimbus
{
	class TriangleMeshEnvironment : public Environment
	{
	public:
		TriangleMeshEnvironment();

		bool Init(const glm::vec3* vertices,
				  const glm::vec3* normals,
				  size_t numVertices,
				  const glm::uvec3* indices,
				  const Face* faces,
				  size_t numFaces,
				  float voxelSize,
				  bool useFaceNormals);

		Type GetType() const override { return Type::TriangleMesh; }
		EnvironmentData GetGpuEnvironmentData() const override;
		uint32_t GetRtPointCount() const override { return m_RtPointCount; }
		const Aabb& GetAabb() const override { return m_Aabb; }

		void ComputeVisibility(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void DetermineLosPaths(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void Transmit(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void Propagate(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void RefineSpecular(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void RefineScatterer(const DeviceBuffer& params, const glm::uvec3& dims) const override;
		void RefineDiffraction(const DeviceBuffer& params, const glm::uvec3& dims) const override;

	private:
		bool ComputeAabb(const glm::vec3* vertices, size_t numVertices);
		bool ComputeVoxelWorld(float voxelSize);
		bool ComputeRayReceptionPoints(const glm::vec3* vertices, const glm::uvec3* indices, const Face* faces, size_t numFaces);
		bool GenerateGpuData(const glm::vec3* vertices,
							 const glm::vec3* normals,
							 size_t numVertices,
							 const glm::uvec3* indices,
							 const Face* faces,
							 size_t numFaces);

	private:
		bool m_UseFaceNormals;
		Aabb m_Aabb;
		VoxelWorldInfo m_VoxelWorldInfo;
		DeviceBuffer m_RtPointsBuffer;
		uint32_t m_RtPointCount;
		DeviceBuffer m_VertexBuffer;
		DeviceBuffer m_NormalBuffer;
		DeviceBuffer m_IndexBuffer;
		DeviceBuffer m_FaceBuffer;
		DeviceBuffer m_VoxelToRtPointIndexMapBuffer;
		AccelerationStructure m_AccelerationStructure;
	};
}