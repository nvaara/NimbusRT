#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <glm/glm.hpp>

#include <iostream>
#include <array>
#include <filesystem>

#include "KernelData.hpp"
#include "VoxelConeTracer.hpp"
#include "InputData.hpp"
#include <glm/gtx/matrix_operation.hpp>
#include <Utils.hpp>

namespace py = pybind11;

using PathData = std::unordered_map<std::string, std::unordered_map<std::string, std::vector<VCT::TraceData>>>;

class Scene
{
public:
	Scene()
	{

	}

	PathData ComputePaths(const VCT::InputData& input,
						  py::array_t<VCT::PointData, py::array::c_style | py::array::forcecast> pointCloud,
						  const std::vector<VCT::Edge>& edges,
					      const std::unordered_map<std::string, VCT::Object3D>& txs,
					      const std::unordered_map<std::string, VCT::Object3D>& rxs) 
	{
		PathData result;
		VCT::VoxelConeTracer coneTracer = VCT::VoxelConeTracer();
		py::buffer_info bufferInfo = pointCloud.request();
		const VCT::PointData* points = static_cast<VCT::PointData*>(bufferInfo.ptr);
		if (coneTracer.Prepare(points, static_cast<size_t>(pointCloud.size()), input, txs, rxs, edges))
		{
			coneTracer.Trace();
			for (uint32_t txID = 0; txID < txs.size(); ++txID)
			{
				const std::string& txName = coneTracer.GetTransmitterName(txID);
				for (uint32_t rxID = 0; rxID < rxs.size(); ++rxID)
				{
					const std::string& rxName = coneTracer.GetReceiverName(rxID);
					coneTracer.Refine(txID, rxID);
					if (auto paths = coneTracer.GetRefinedPathStorage().GetPaths(txID, rxID))
					{
						result[txName][rxName] = *paths;
					}
				}
			}
		}
		return result;
	}
};


PYBIND11_MODULE(_C, m)
{
	auto os = py::module::import("os");
	py::object cwd = os.attr("getcwd")();
	std::filesystem::current_path(std::string(py::str(cwd)));

	if (!VCT::KernelData::Initialize())
		throw std::runtime_error("Failed to load GPU Device or kernels.");
	
	Py_AtExit([]() { VCT::KernelData::Destroy(); });

	m.doc() = "NimbusRT native code module.";
	
	PYBIND11_NUMPY_DTYPE(VCT::PointData, position.x, position.y, position.z, normal.x, normal.y, normal.z, label, material);

	auto iaEnum = py::enum_<VCT::InteractionType>(m, "NativeInteractionType")
		.value("DIFFRACTION", VCT::InteractionType::Diffraction)
		.value("REFLECTION", VCT::InteractionType::Reflection);

	auto interactions = py::class_<VCT::Interaction>(m, "NativeInteraction")
		.def_readwrite("label", &VCT::Interaction::label)
		.def_readwrite("type", &VCT::Interaction::type)
		.def_readwrite("position", &VCT::Interaction::posArr)
		.def_readwrite("normal", &VCT::Interaction::normArr)
		.def_readwrite("materialID", &VCT::Interaction::materialID);

	auto traceData = py::class_<VCT::TraceData>(m, "NativeTraceData")
		.def_readwrite("transmitterID", &VCT::TraceData::transmitterID)
		.def_readwrite("receiverID", &VCT::TraceData::receiverID)
		.def_readwrite("num_interactions", &VCT::TraceData::numInteractions)
		.def_readwrite("time_delay", &VCT::TraceData::timeDelay)
		.def_readwrite("interactions", &VCT::TraceData::interactions);

	auto edge = py::class_<VCT::Edge>(m, "NativeEdge")
		.def(py::init<const VCT::V3&,
					  const VCT::V3&,
					  const VCT::V3&,
					  const VCT::V3&>());

	auto scene = py::class_<Scene>(m, "NativeScene")
		.def(py::init<>())
		.def("_compute_paths", &Scene::ComputePaths);

	auto sceneSettings = py::class_<VCT::SceneSettings>(m, "SceneSettings")
		.def(py::init<>())
		.def_readwrite("frequency", &VCT::SceneSettings::frequency)
		.def_readwrite("voxel_size", &VCT::SceneSettings::voxelSize)
		.def_readwrite("voxel_division_factor", &VCT::SceneSettings::voxelDivisionFactor)
		.def_readwrite("subvoxel_division_factor", &VCT::SceneSettings::subvoxelDivisionFactor)
		.def_readwrite("received_path_buffer_size", &VCT::SceneSettings::receivedPathBufferSize)
		.def_readwrite("propagation_path_buffer_size", &VCT::SceneSettings::propagationPathBufferSize)
		.def_readwrite("propagation_buffer_size_increase_factor", &VCT::SceneSettings::propagationBufferSizeIncreaseFactor)
		.def_readwrite("sample_radius_coarse", &VCT::SceneSettings::sampleRadiusCoarse)
		.def_readwrite("sample_radius_refine", &VCT::SceneSettings::sampleRadiusRefine)
		.def_readwrite("variance_factor_coarse", &VCT::SceneSettings::varianceFactorCoarse)
		.def_readwrite("variance_factor_refine", &VCT::SceneSettings::varianceFactorRefine)
		.def_readwrite("sdf_threshold_coarse", &VCT::SceneSettings::sdfThresholdCoarse)
		.def_readwrite("sdf_threshold_refine", &VCT::SceneSettings::sdfThresholdRefine)
		.def_readwrite("num_iterations", &VCT::SceneSettings::numIterations)
		.def_readwrite("delta", &VCT::SceneSettings::delta)
		.def_readwrite("alpha", &VCT::SceneSettings::alpha)
		.def_readwrite("beta", &VCT::SceneSettings::beta)
		.def_readwrite("angle_threshold", &VCT::SceneSettings::angleThreshold)
		.def_readwrite("distance_threshold", &VCT::SceneSettings::distanceThreshold)
		.def_readwrite("block_size", &VCT::SceneSettings::blockSize)
		.def_readwrite("num_coarse_paths_per_unique_route", &VCT::SceneSettings::numCoarsePathsPerUniqueRoute);

	auto object = py::class_<VCT::Object3D>(m, "NativeObject3D")
		.def(py::init<const std::array<float, 3>&>());

	auto cpInput = py::class_<VCT::InputData>(m, "InputData")
		.def(py::init<>())
		.def_readwrite("scene_settings", &VCT::InputData::sceneSettings)
		.def_readwrite("num_interactions", &VCT::InputData::numInteractions)
		.def_readwrite("num_diffractions", &VCT::InputData::numDiffractions);
}