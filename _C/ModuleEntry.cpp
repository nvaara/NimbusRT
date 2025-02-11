#include "Scene.hpp"
#include "Nimbus/KernelData.hpp"
#include <filesystem>

PYBIND11_MODULE(_C, m)
{
	auto os = py::module::import("os");
	py::object cwd = os.attr("getcwd")();
	std::filesystem::current_path(std::string(py::str(cwd)));

	if (!Nimbus::KernelData::Initialize())
		throw std::runtime_error("Failed to load GPU Device or kernels.");
	
	Py_AtExit([]() { Nimbus::KernelData::Destroy(); });

	m.doc() = "NimbusRT Python Interface.";
	
	PYBIND11_NUMPY_DTYPE(Nimbus::PointData, position.x, position.y, position.z, normal.x, normal.y, normal.z, label, material);
	PYBIND11_NUMPY_DTYPE(Nimbus::EdgeData, start.x, start.y, start.z, end.x, end.y, end.z, normal1.x, normal1.y, normal1.z, normal2.x, normal2.y, normal2.z, material1, material2);
	PYBIND11_NUMPY_DTYPE(Nimbus::Face, normal.x, normal.y, normal.z, label, material);
	PYBIND11_NUMPY_DTYPE(glm::vec3, x, y, z);
	PYBIND11_NUMPY_DTYPE(glm::uvec3, x, y, z);

	py::class_<Scene>(m, "NativeScene")
		.def(py::init<>())
		.def("_compute_path_data", &Scene::ComputePathData)
		.def("_compute_sionna_path_data", &Scene::ComputeSionnaPathData)
		.def("_set_point_cloud", &Scene::SetPointCloud)
		.def("_set_triangle_mesh", &Scene::SetTriangleMesh)
		.def("_compute_cm_st", &Scene::ComputeCoverageMap);

	py::class_<PathWrapper>(m, "NativePathData")
		.def_property_readonly("transmitters", &PathWrapper::GetTransmitters)
		.def_property_readonly("receivers", &PathWrapper::GetReceivers)
		.def_property_readonly("interactions", &PathWrapper::GetInteractions)
		.def_property_readonly("normals", &PathWrapper::GetNormals)
		.def_property_readonly("label_ids", &PathWrapper::GetLabels)
		.def_property_readonly("material_ids", &PathWrapper::GetMaterials)
	    .def_property_readonly("time_delays", &PathWrapper::GetTimeDelays)
		.def_property_readonly("tx_ids", &PathWrapper::GetTxIDs)
		.def_property_readonly("rx_ids", &PathWrapper::GetRxIDs)
		.def_property_readonly("path_types", &PathWrapper::GetPathTypes)
		.def_property_readonly("num_interactions", &PathWrapper::GetNumInteractions);

	py::enum_<Nimbus::PathType>(m, "NativePathType")
		.value("LINE_OF_SIGHT", Nimbus::PathType::LineOfSight)
		.value("SPECULAR", Nimbus::PathType::Specular)
		.value("SCATTERING", Nimbus::PathType::Scattering)
		.value("DIFFRACTION", Nimbus::PathType::Diffraction);

	py::class_<CoverageWrapper>(m, "NativeCoverageMapData")
		.def_property_readonly("transmitters", &PathWrapper::GetTransmitters)
		.def_property_readonly("receivers", &PathWrapper::GetReceivers)
		.def_property_readonly("interactions", &PathWrapper::GetInteractions)
		.def_property_readonly("normals", &PathWrapper::GetNormals)
		.def_property_readonly("label_ids", &PathWrapper::GetLabels)
		.def_property_readonly("material_ids", &PathWrapper::GetMaterials)
		.def_property_readonly("time_delays", &PathWrapper::GetTimeDelays)
		.def_property_readonly("tx_ids", &PathWrapper::GetTxIDs)
		.def_property_readonly("rx_ids", &PathWrapper::GetRxIDs)
		.def_property_readonly("path_types", &PathWrapper::GetPathTypes)
		.def_property_readonly("num_interactions", &PathWrapper::GetNumInteractions)
		.def_property_readonly("shape", &CoverageWrapper::GetDimensions)
		.def_property_readonly("rx_coords_2d", &CoverageWrapper::GetRx2D);


	py::class_<SionnaPathWrapper>(m, "NativeSionnaPathData")
		.def_property_readonly("sources", &SionnaPathWrapper::GetTransmitters)
		.def_property_readonly("targets", &SionnaPathWrapper::GetReceivers)
		.def("vertices", &SionnaPathWrapper::GetInteractions)
		.def("normals", &SionnaPathWrapper::GetNormals)
		.def("objects", &SionnaPathWrapper::GetMaterials)
		.def("mask", &SionnaPathWrapper::GetMask)
		.def("theta_t", &SionnaPathWrapper::GetAodElevation)
		.def("theta_r", &SionnaPathWrapper::GetAoaElevation)
		.def("phi_t", &SionnaPathWrapper::GetAodAzimuth)
		.def("phi_r", &SionnaPathWrapper::GetAoaAzimuth)
		.def("tau", &SionnaPathWrapper::GetTimeDelays)
		.def("total_distance", &SionnaPathWrapper::GetTotalDistance)
		.def("k_tx", &SionnaPathWrapper::GetKTx)
		.def("k_rx", &SionnaPathWrapper::GetKRx)
		.def("k_i", &SionnaPathWrapper::GetIncidentRays)
		.def("k_r", &SionnaPathWrapper::GetDeflectedRays)
		.def("scat_last_objects", &SionnaPathWrapper::GetScatLastObjects)
		.def("scat_last_vertices", &SionnaPathWrapper::GetScatLastVertices)
		.def("scat_last_k_i", &SionnaPathWrapper::GetScatLastIncident)
		.def("scat_k_s", &SionnaPathWrapper::GetScatLastDeflected)
		.def("scat_last_normals", &SionnaPathWrapper::GetScatLastNormals)
		.def("scat_src_2_last_int_dist", &SionnaPathWrapper::GetScatDistToLastIa)
		.def("scat_2_target_dist", &SionnaPathWrapper::GetScatDistFromLastIaToRx)
		.def("cos_theta_i", &SionnaPathWrapper::GetCosThetaTx)
		.def("cos_theta_m", &SionnaPathWrapper::GetCosThetaRx)
		.def("distance_tx_ris", &SionnaPathWrapper::GetDistanceTxRis)
		.def("distance_rx_ris", &SionnaPathWrapper::GetDistanceRxRis)
		.def("max_link_paths", &SionnaPathWrapper::GetMaxLinkPaths);

	py::class_<Nimbus::ScatterTracingParams>(m, "NativeRTParams")
		.def(py::init<>())
		.def_readwrite("_max_depth", &Nimbus::ScatterTracingParams::maxNumInteractions)
		.def_readwrite("_los", &Nimbus::ScatterTracingParams::los)
		.def_readwrite("_reflection", &Nimbus::ScatterTracingParams::reflection)
		.def_readwrite("_scattering", &Nimbus::ScatterTracingParams::scattering)
		.def_readwrite("_diffraction", &Nimbus::ScatterTracingParams::diffraction)
		.def_readwrite("_ris", &Nimbus::ScatterTracingParams::ris)
		.def_readwrite("_refine_max_iterations", &Nimbus::ScatterTracingParams::numRefineIterations)
		.def_readwrite("_refine_max_correction_iterations", &Nimbus::ScatterTracingParams::refineMaxCorrectionIterations)
		.def_readwrite("_refine_convergence_threshold", &Nimbus::ScatterTracingParams::refineDelta)
		.def_readwrite("_refine_beta", &Nimbus::ScatterTracingParams::refineBeta)
		.def_readwrite("_refine_alpha", &Nimbus::ScatterTracingParams::refineAlpha)
		.def_readwrite("_refine_angle_degrees_threshold", &Nimbus::ScatterTracingParams::refineAngleDegreesThreshold)
		.def_readwrite("_refine_distance_threshold", &Nimbus::ScatterTracingParams::refineDistanceThreshold)
		.def_readwrite("_ray_bias", &Nimbus::ScatterTracingParams::rayBias);

	py::class_<RisWrapper>(m, "NativeRisData")
		.def(py::init<>())
		.def_readwrite("cell_world_positions", &RisWrapper::cellWorldPositions)
		.def_readwrite("object_ids", &RisWrapper::objectIds)
		.def_readwrite("cell_object_ids", &RisWrapper::cellObjectIds)
		.def_readwrite("normals", &RisWrapper::normals)
		.def_readwrite("centers", &RisWrapper::centers)
		.def_readwrite("size", &RisWrapper::size);
}