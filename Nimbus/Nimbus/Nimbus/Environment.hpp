#pragma once
#include <Nimbus/Types.hpp>
#include <Nimbus/DeviceBuffer.hpp>
#include <Nimbus/CudaUtils.hpp>

namespace Nimbus
{
	class Environment
	{
	public:
		enum class Type
		{
			None,
			PointCloud,
			TriangleMesh
		};

		virtual ~Environment() = default;
		
		virtual float GetVoxelSize() const = 0;
		virtual Type GetType() const = 0;
		virtual EnvironmentData GetGpuEnvironmentData() = 0;
		virtual uint32_t GetRtPointCount() const = 0;
		virtual const Aabb& GetAabb() const = 0;
		const std::vector<DiffractionEdge> GetEdges() const { return m_Edges; };
		bool HasEdges() const { return GetEdges().size() > 0u; }
		uint32_t GetEdgeCount() const { return static_cast<uint32_t>(GetEdges().size()); }
		uint32_t GetRisPointCount() const { return m_RisData.cellCount; };
		glm::vec3 GetCenter() const;
		glm::vec3 GetSceneSize() const;
		bool InitRisGasData(const RisData& risData);

		virtual void ComputeVisibility(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void DetermineLosPaths(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void Transmit(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void Propagate(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void RefineSpecular(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void RefineScatterer(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void RefineDiffraction(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void ComputeRISPaths(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;

	protected:
		struct RisGasData
		{
			DeviceBuffer vertexBuffer;
			DeviceBuffer indexBuffer;
			DeviceBuffer objectIds;
			DeviceBuffer cellObjectIds;
			DeviceBuffer cellWorldPositions;
			DeviceBuffer normals;
			AccelerationStructure gas;
			uint32_t cellCount;
		};

	protected:
		Environment() = default;
		OptixTraversableHandle GetAccelerationStructure();
		bool ProcessEdges(const EdgeData* edges, size_t numEdges);

	protected:
		std::vector<DiffractionEdge> m_Edges;
		DeviceBuffer m_EdgeBuffer;
		AccelerationStructure m_AccelerationStructure;
		RisGasData m_RisData;
		DeviceBuffer m_InstanceBuffer;
		AccelerationStructure m_InstanceAs;
	};
}