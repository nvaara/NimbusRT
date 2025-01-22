#include "Scene.hpp"
#include "Nimbus/ScatterTracer.hpp"
#include "Nimbus/Profiler.hpp"

void Scene::SetPointCloud(py::array_t<Nimbus::PointData, py::array::c_style | py::array::forcecast> pointCloud,
						  std::optional<py::array_t<Nimbus::EdgeData, py::array::c_style | py::array::forcecast>> edges,
						  float voxelSize,
						  float aabbBias)
{
	auto env = std::make_unique<Nimbus::PointCloudEnvironment>();
	const Nimbus::EdgeData* edgePtr = edges ? edges.value().data() : nullptr;
	size_t numEdges = edges ? edges.value().size() : 0u;
	
	if (env->Init(pointCloud.data(), pointCloud.size(), edgePtr, numEdges, voxelSize, aabbBias))
	{
		m_Environment = std::move(env);
		return;
	}
	throw std::runtime_error("Failed to set point cloud.");
}

void Scene::SetTriangleMesh(py::array_t<glm::vec3, py::array::c_style | py::array::forcecast> vertices,
						    py::array_t<glm::vec3, py::array::c_style | py::array::forcecast> normals,
						    py::array_t<glm::uvec3, py::array::c_style | py::array::forcecast> indices,
						    py::array_t<Nimbus::Face, py::array::c_style | py::array::forcecast> faces,
						    float voxelSize,
							bool useFaceNormals)
{
	auto env = std::make_unique<Nimbus::TriangleMeshEnvironment>();
	if (env->Init(vertices.data(), normals.data(), vertices.size(), indices.data(), faces.data(), indices.size(), voxelSize, useFaceNormals))
	{
		m_Environment = std::move(env);
		return;
	}
	throw std::runtime_error("Failed to set triangle mesh.");
}

std::unique_ptr<PathWrapper> Scene::ComputePathData(const Nimbus::ScatterTracingParams& params,
													const py::array_t<float, py::array::c_style | py::array::forcecast>& txs,
													const py::array_t<float, py::array::c_style | py::array::forcecast>& rxs)
{
	Nimbus::ScatterTracer st = Nimbus::ScatterTracer();
	const glm::vec3* txPtr = reinterpret_cast<const glm::vec3*>(txs.data());
	const glm::vec3* rxPtr = reinterpret_cast<const glm::vec3*>(rxs.data());
	if (m_Environment && st.Prepare(*m_Environment, params, txPtr, static_cast<uint32_t>(txs.shape(0)), rxPtr, static_cast<uint32_t>(rxs.shape(0))))
	{
		PROFILE_SCOPE();
		return std::make_unique<PathWrapper>(st.Trace());
	}
	throw std::runtime_error("Failed to compute path data.");
	return {};
}

std::unique_ptr<CoverageWrapper> Scene::ComputeCoverageMap(const Nimbus::ScatterTracingParams& params,
														   const py::array_t<float, py::array::c_style | py::array::forcecast>& tx,
														   float size,
														   float height)
{
	Nimbus::ScatterTracer st = Nimbus::ScatterTracer();
	Nimbus::CoverageMapInfo mapInfo{};
	std::vector<glm::vec3> receivers;
	if (m_Environment && st.CreateCoverageMapInfo(*m_Environment, *reinterpret_cast<const glm::vec3*>(tx.data()), size, height, mapInfo, receivers))
	{
		if (st.Prepare(*m_Environment, params, reinterpret_cast<const glm::vec3*>(tx.data()), 1u, receivers.data(), static_cast<uint32_t>(receivers.size())))
		{
			PROFILE_SCOPE();
			return std::make_unique<CoverageWrapper>(st.Trace(), std::move(mapInfo));
		}
	}
	throw std::runtime_error("Failed to compute coverage map.");
	return {};
}

std::unique_ptr<SionnaPathWrapper> Scene::ComputeSionnaPathData(const Nimbus::ScatterTracingParams& params,
															    const py::array_t<float, py::array::c_style | py::array::forcecast>& txs,
															    const py::array_t<float, py::array::c_style | py::array::forcecast>& rxs)
{
	Nimbus::ScatterTracer st = Nimbus::ScatterTracer();
	const glm::vec3* txPtr = reinterpret_cast<const glm::vec3*>(txs.data());
	const glm::vec3* rxPtr = reinterpret_cast<const glm::vec3*>(rxs.data());
	if (m_Environment && st.Prepare(*m_Environment, params, txPtr, static_cast<uint32_t>(txs.shape(0)), rxPtr, static_cast<uint32_t>(rxs.shape(0))))
	{
		PROFILE_SCOPE();
		return std::make_unique<SionnaPathWrapper>(m_Environment->GetVoxelSize(), st.Trace());
	}
	throw std::runtime_error("Failed to compute path data.");
	return {};
}
