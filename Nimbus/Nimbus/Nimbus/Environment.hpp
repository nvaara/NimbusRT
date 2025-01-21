#pragma once
#include <Nimbus/Types.hpp>

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
		virtual Type GetType() const = 0;
		virtual EnvironmentData GetGpuEnvironmentData() const = 0;
		virtual uint32_t GetRtPointCount() const = 0;
		virtual const Aabb& GetAabb() const = 0;

		virtual void ComputeVisibility(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void DetermineLosPaths(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void Transmit(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void Propagate(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void RefineSpecular(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void RefineScatterer(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;
		virtual void RefineDiffraction(const DeviceBuffer& params, const glm::uvec3& dims) const = 0;

	protected:
		Environment() = default;
	};
}