#include "Wrappers.hpp"

PathWrapper::PathWrapper(std::unique_ptr<Nimbus::PathStorage>&& path)
	: m_Data(path->ToPathData())
{

}

py::array_t<float, py::array::c_style> PathWrapper::GetTransmitters() const
{
	std::array<size_t, 2> shape = { m_Data.transmitters.size(), glm::vec3::length() };
	std::array<size_t, 2> stride = { shape[1] * sizeof(float), sizeof(float), };
	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_Data.transmitters.data()), py::none());
}

py::array_t<float, py::array::c_style> PathWrapper::GetReceivers() const
{
	std::array<size_t, 2> shape = { m_Data.receivers.size(), glm::vec3::length() };
	std::array<size_t, 2> stride = { shape[1] * sizeof(float), sizeof(float), };
	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_Data.receivers.data()), py::none());
}

py::array_t<float, py::array::c_style> PathWrapper::GetInteractions() const
{
	std::array<size_t, 3> shape = { m_Data.maxNumIa, m_Data.interactions.size() / glm::max(m_Data.maxNumIa, 1u), glm::vec3::length() };
	std::array<size_t, 3> stride = { shape[1] * sizeof(glm::vec3), sizeof(glm::vec3), sizeof(float) };
	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_Data.interactions.data()), py::none());
}

py::array_t<float, py::array::c_style> PathWrapper::GetNormals() const
{
	std::array<size_t, 3> shape = { m_Data.maxNumIa, m_Data.normals.size() / glm::max(m_Data.maxNumIa, 1u), glm::vec3::length() };
	std::array<size_t, 3> stride = { shape[1] * sizeof(glm::vec3), sizeof(glm::vec3), sizeof(float) };
	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_Data.normals.data()), py::none());
}

py::array_t<uint32_t, py::array::c_style> PathWrapper::GetLabels() const
{
	std::array<size_t, 2> shape = { m_Data.maxNumIa, m_Data.labels.size() / glm::max(m_Data.maxNumIa, 1u) };
	std::array<size_t, 2> stride = { shape[1] * sizeof(uint32_t), sizeof(uint32_t), };
	return py::array_t<uint32_t, py::array::c_style>(shape, stride, m_Data.labels.data(), py::none());
}

py::array_t<uint32_t, py::array::c_style> PathWrapper::GetMaterials() const
{
	std::array<size_t, 2> shape = { m_Data.maxNumIa, m_Data.materials.size() / glm::max(m_Data.maxNumIa, 1u) };
	std::array<size_t, 2> stride = { shape[1] * sizeof(uint32_t), sizeof(uint32_t), };
	return py::array_t<uint32_t, py::array::c_style>(shape, stride, m_Data.materials.data(), py::none());
}

py::array_t<double, py::array::c_style> PathWrapper::GetTimeDelays() const
{
	return py::array_t<double, py::array::c_style>({ m_Data.timeDelays.size() }, { sizeof(double) }, m_Data.timeDelays.data(), py::none());
}

py::array_t<uint32_t, py::array::c_style> PathWrapper::GetTxIDs() const
{
	return py::array_t<uint32_t, py::array::c_style>({ m_Data.txIDs.size() }, { sizeof(uint32_t) }, m_Data.txIDs.data(), py::none());
}

py::array_t<uint32_t, py::array::c_style> PathWrapper::PathWrapper::GetRxIDs() const
{
	return py::array_t<uint32_t, py::array::c_style>({ m_Data.rxIDs.size() }, { sizeof(uint32_t) }, m_Data.rxIDs.data(), py::none());
}

py::array_t<Nimbus::PathType, py::array::c_style> PathWrapper::GetPathTypes() const
{
	return py::array_t<Nimbus::PathType, py::array::c_style>({ m_Data.pathTypes.size() }, { sizeof(uint8_t) }, m_Data.pathTypes.data(), py::none());
}

py::array_t<uint8_t, py::array::c_style> PathWrapper::GetNumInteractions() const
{
	return py::array_t<uint8_t, py::array::c_style>({ m_Data.numInteractions.size() }, { sizeof(uint8_t) }, m_Data.numInteractions.data(), py::none());
}

CoverageWrapper::CoverageWrapper(std::unique_ptr<Nimbus::PathStorage>&& path, Nimbus::CoverageMapInfo&& mapInfo)
	: PathWrapper(std::move(path))
	, m_CoverageMapInfo(std::move(mapInfo))
{

}

py::array_t<uint32_t, py::array::c_style> CoverageWrapper::GetRx2D() const
{
	std::array<size_t, 2> shape = { m_CoverageMapInfo.rxCoords2D.size(), glm::uvec2::length() };
	std::array<size_t, 2> stride = { shape[1] * sizeof(uint32_t), sizeof(uint32_t), };
	return py::array_t<uint32_t, py::array::c_style>(shape, stride, reinterpret_cast<const uint32_t*>(m_CoverageMapInfo.rxCoords2D.data()), py::none());
}

py::array_t<uint32_t, py::array::c_style> CoverageWrapper::GetDimensions() const
{
	return py::array_t<uint32_t, py::array::c_style>({ glm::uvec2::length() }, { sizeof(glm::uvec2::value_type) }, &m_CoverageMapInfo.dimensions.x, py::none());
}

SionnaPathWrapper::SionnaPathWrapper(const Nimbus::Environment& env, std::unique_ptr<Nimbus::PathStorage>&& path)
	: m_SionnaData(path->ToSionnaPathData(env))
{

}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetTransmitters() const
{
	std::array<size_t, 2> shape = { m_SionnaData.transmitters.size(), glm::vec3::length() };
	std::array<size_t, 2> stride = { sizeof(glm::vec3), sizeof(float)};
	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.transmitters.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetReceivers() const
{
	std::array<size_t, 2> shape = { m_SionnaData.receivers.size(), glm::vec3::length() };
	std::array<size_t, 2> stride = { sizeof(glm::vec3), sizeof(float) };
	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.receivers.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetInteractions(uint32_t sionnaPathType) const
{
	std::array<size_t, 5> shape = { m_SionnaData.maxNumIa[sionnaPathType], m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType], glm::vec3::length()};
	std::array<size_t, 5> stride = { shape[1] * shape[2] * shape[3] * sizeof(glm::vec3),
									 shape[2] * shape[3] * sizeof(glm::vec3),
									 shape[3] * sizeof(glm::vec3),
									 sizeof(glm::vec3),
									 sizeof(float)};

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].interactions.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetNormals(uint32_t sionnaPathType) const
{
	std::array<size_t, 5> shape = { m_SionnaData.maxNumIa[sionnaPathType], m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType], glm::vec3::length() };
	std::array<size_t, 5> stride = { shape[1] * shape[2] * shape[3] * sizeof(glm::vec3),
									 shape[2] * shape[3] * sizeof(glm::vec3),
									 shape[3] * sizeof(glm::vec3),
									 sizeof(glm::vec3),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].normals.data()), py::none());
}

py::array_t<int32_t, py::array::c_style> SionnaPathWrapper::GetMaterials(uint32_t sionnaPathType) const
{
	std::array<size_t, 4> shape = { m_SionnaData.maxNumIa[sionnaPathType], m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType] };
	std::array<size_t, 4> stride = { shape[1] * shape[2] * shape[3] * sizeof(uint32_t),
									 shape[2] * shape[3] * sizeof(uint32_t),
									 shape[3] * sizeof(uint32_t),
									 sizeof(uint32_t) };
	
	return py::array_t<int32_t, py::array::c_style>(shape, stride, reinterpret_cast<const int32_t*>(m_SionnaData.paths[sionnaPathType].materials.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetIncidentRays(uint32_t sionnaPathType) const
{
	std::array<size_t, 5> shape = { m_SionnaData.maxNumIa[sionnaPathType] + 1, m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType], glm::vec3::length() };
	std::array<size_t, 5> stride = { shape[1] * shape[2] * shape[3] * sizeof(glm::vec3),
									 shape[2] * shape[3] * sizeof(glm::vec3),
									 shape[3] * sizeof(glm::vec3),
									 sizeof(glm::vec3),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].incidentRays.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetDeflectedRays(uint32_t sionnaPathType) const
{
	std::array<size_t, 5> shape = { m_SionnaData.maxNumIa[sionnaPathType], m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType], glm::vec3::length()};
	std::array<size_t, 5> stride = { shape[1] * shape[2] * shape[3] * sizeof(glm::vec3),
									 shape[2] * shape[3] * sizeof(glm::vec3),
									 shape[3] * sizeof(glm::vec3),
									 sizeof(glm::vec3),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].deflectedRays.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetTimeDelays(uint32_t sionnaPathType) const
{
	std::array<size_t, 3> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType] };
	std::array<size_t, 3> stride = { shape[1] * shape[2] * sizeof(float),
									 shape[2] * sizeof(float),
									 sizeof(float) };
	
	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].timeDelays.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetTotalDistance(uint32_t sionnaPathType) const
{
	std::array<size_t, 3> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType] };
	std::array<size_t, 3> stride = { shape[1] * shape[2] * sizeof(float),
									 shape[2] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].totalDistance.data()), py::none());
}

py::array_t<uint8_t, py::array::c_style> SionnaPathWrapper::GetMask(uint32_t sionnaPathType) const
{
	std::array<size_t, 3> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType] };
	std::array<size_t, 3> stride = { shape[1] * shape[2] * sizeof(uint8_t),
									 shape[2] * sizeof(uint8_t),
									 sizeof(uint8_t) };

	return py::array_t<uint8_t, py::array::c_style>(shape, stride, reinterpret_cast<const uint8_t*>(m_SionnaData.paths[sionnaPathType].mask.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetKTx(uint32_t sionnaPathType) const
{
	std::array<size_t, 4> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType], glm::vec3::length()};
	std::array<size_t, 4> stride = { shape[1] * shape[2] * shape[3] * sizeof(float),
									 shape[2] * shape[3] * sizeof(float),
									 shape[3] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].kTx.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetKRx(uint32_t sionnaPathType) const
{
	std::array<size_t, 4> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType], glm::vec3::length()};
	std::array<size_t, 4> stride = { shape[1] * shape[2] * shape[3] * sizeof(float),
									 shape[2] * shape[3] * sizeof(float),
									 shape[3] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].kRx.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetAodElevation(uint32_t sionnaPathType) const
{
	std::array<size_t, 3> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType]};
	std::array<size_t, 3> stride = { shape[1] * shape[2] * sizeof(float),
									 shape[2] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].aodElevation.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetAodAzimuth(uint32_t sionnaPathType) const
{
	std::array<size_t, 3> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType] };
	std::array<size_t, 3> stride = { shape[1] * shape[2] * sizeof(float),
									 shape[2] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].aodAzimuth.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetAoaElevation(uint32_t sionnaPathType) const
{
	std::array<size_t, 3> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType]};
	std::array<size_t, 3> stride = { shape[1] * shape[2] * sizeof(float),
									 shape[2] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].aoaElevation.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetAoaAzimuth(uint32_t sionnaPathType) const
{
	std::array<size_t, 3> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType] };
	std::array<size_t, 3> stride = { shape[1] * shape[2] * sizeof(float),
									 shape[2] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].aoaAzimuth.data()), py::none());
}

py::array_t<int32_t, py::array::c_style> SionnaPathWrapper::GetScatLastObjects(uint32_t sionnaPathType) const
{
	std::array<size_t, 3> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType] };
	std::array<size_t, 3> stride = { shape[1] * shape[2] * sizeof(uint32_t),
									 shape[2] * sizeof(uint32_t),
									 sizeof(uint32_t) };

	return py::array_t<int32_t, py::array::c_style>(shape, stride, reinterpret_cast<const int32_t*>(m_SionnaData.paths[sionnaPathType].scattering.lastObjects.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetScatLastVertices(uint32_t sionnaPathType) const
{
	std::array<size_t, 4> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType], glm::vec3::length() };
	std::array<size_t, 4> stride = { shape[1] * shape[2] * shape[3] * sizeof(float),
									 shape[2] * shape[3] * sizeof(float),
									 shape[3] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].scattering.lastVertices.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetScatLastIncident(uint32_t sionnaPathType) const
{
	std::array<size_t, 4> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType], glm::vec3::length() };
	std::array<size_t, 4> stride = { shape[1] * shape[2] * shape[3] * sizeof(float),
									 shape[2] * shape[3] * sizeof(float),
									 shape[3] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].scattering.lastIncident.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetScatLastDeflected(uint32_t sionnaPathType) const
{
	std::array<size_t, 4> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType], glm::vec3::length() };
	std::array<size_t, 4> stride = { shape[1] * shape[2] * shape[3] * sizeof(float),
									 shape[2] * shape[3] * sizeof(float),
									 shape[3] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].scattering.lastDeflected.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetScatLastNormals(uint32_t sionnaPathType) const
{
	std::array<size_t, 4> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType], glm::vec3::length() };
	std::array<size_t, 4> stride = { shape[1] * shape[2] * shape[3] * sizeof(float),
									 shape[2] * shape[3] * sizeof(float),
									 shape[3] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].scattering.lastNormal.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetScatDistToLastIa(uint32_t sionnaPathType) const
{
	std::array<size_t, 3> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType] };
	std::array<size_t, 3> stride = { shape[1] * shape[2] * sizeof(float),
									 shape[2] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].scattering.distToLastIa.data()), py::none());
}

py::array_t<float, py::array::c_style> SionnaPathWrapper::GetScatDistFromLastIaToRx(uint32_t sionnaPathType) const
{
	std::array<size_t, 3> shape = { m_SionnaData.receivers.size(), m_SionnaData.transmitters.size(), m_SionnaData.maxLinkPaths[sionnaPathType] };
	std::array<size_t, 3> stride = { shape[1] * shape[2] * sizeof(float),
									 shape[2] * sizeof(float),
									 sizeof(float) };

	return py::array_t<float, py::array::c_style>(shape, stride, reinterpret_cast<const float*>(m_SionnaData.paths[sionnaPathType].scattering.distFromLastIaToRx.data()), py::none());
}

int32_t SionnaPathWrapper::GetMaxLinkPaths(uint32_t sionnaPathType) const
{
	return static_cast<int32_t>(m_SionnaData.maxLinkPaths[sionnaPathType]);
}
