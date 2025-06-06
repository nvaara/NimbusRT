#pragma once
#include "Wrappers.hpp"
#include "Nimbus/PointCloudEnvironment.hpp"
#include "Nimbus/TriangleMeshEnvironment.hpp"
#include <array>

class Scene
{
public:
	Scene() = default;

	void SetPointCloud(py::array_t<Nimbus::PointData, py::array::c_style | py::array::forcecast> pointCloud,
					   std::optional<py::array_t<Nimbus::EdgeData, py::array::c_style | py::array::forcecast>> edges,
					   float voxelSize,
					   float pointRadius,
					   float sdfThreshold,
					   float lambdaDistance);

	void SetTriangleMesh(py::array_t<glm::vec3, py::array::c_style | py::array::forcecast> vertices,
					     py::array_t<glm::vec3, py::array::c_style | py::array::forcecast> normals,
					     py::array_t<glm::uvec3, py::array::c_style | py::array::forcecast> indices,
					     py::array_t<Nimbus::Face, py::array::c_style | py::array::forcecast> faceData,
						 std::optional<py::array_t<Nimbus::EdgeData, py::array::c_style | py::array::forcecast>> edges,
					     float voxelSize,
						 bool useFaceNormals);

	std::unique_ptr<PathWrapper> ComputePathData(const Nimbus::ScatterTracingParams& params,
												 const py::array_t<float, py::array::c_style | py::array::forcecast>& txs,
												 const py::array_t<float, py::array::c_style | py::array::forcecast>& rxs);

	std::unique_ptr<CoverageWrapper> ComputeCoverageMap(const Nimbus::ScatterTracingParams& params,
													    const py::array_t<float, py::array::c_style | py::array::forcecast>& tx,
													    float size,
													    float height);

	std::unique_ptr<SionnaPathWrapper> ComputeSionnaPathData(const Nimbus::ScatterTracingParams& params,
															 const py::array_t<float, py::array::c_style | py::array::forcecast>& txs,
															 const py::array_t<float, py::array::c_style | py::array::forcecast>& rxs,
															 const RisWrapper& risWrapper);

	std::unique_ptr<SionnaCoverageWrapper> ComputeSionnaCoverageMap(const Nimbus::ScatterTracingParams& params,
																    const py::array_t<float, py::array::c_style | py::array::forcecast>& txs,
																    float size,
																    float height,
																    const RisWrapper& risWrapper);
	
	std::array<float, 3> GetSize() const;
	std::array<float, 3> GetCenter() const;

private:
	std::unique_ptr<Nimbus::Environment> m_Environment;
};