#pragma once
#include "Logger.hpp"
#include "TriangleMeshEnvironment.hpp"
#include "Nimbus/Utils.hpp"
#include "KernelData.hpp"
#include "Profiler.hpp"

namespace Nimbus
{
	TriangleMeshEnvironment::TriangleMeshEnvironment()
		: m_VoxelSize(0.0f)
		, m_Aabb()
		, m_VoxelWorldInfo()
		, m_RtPointCount(0u)
		, m_UseFaceNormals(false)
	{

	}

	bool TriangleMeshEnvironment::Init(const glm::vec3* vertices,
									   const glm::vec3* normals,
									   size_t numVertices,
									   const glm::uvec3* indices,
									   const Face* faces,
									   size_t numFaces,
									   const EdgeData* edges,
									   size_t numEdges,
									   float voxelSize,
									   bool useFaceNormals)
	{
		m_VoxelSize = voxelSize;
		m_UseFaceNormals = useFaceNormals;

		if (!ComputeAabb(vertices, numVertices))
		{
			LOG("Failed to compute aabb. Vertex count is 0?");
			return false;
		}
		if (!ComputeVoxelWorld(voxelSize))
		{
			LOG("Failed to compute voxel world. Voxel size is 0 or voxel world dimensions are 0.");
			return false;
		}

		if (!GenerateGpuData(vertices, normals, numVertices, indices, faces, numFaces))
		{
			LOG("Failed to initialize ray tracing data.");
			return false;
		}

		if (!ComputeRayReceptionPoints(vertices, indices, faces, numFaces))
		{
			LOG("Failed to compute ray reception points from the model.");
			return false;
		}

		if (edges && !ProcessEdges(edges, numEdges))
		{
			LOG("Failed to process edges.");
			return false;
		}

		return true;
	}

	EnvironmentData TriangleMeshEnvironment::GetGpuEnvironmentData()
	{
		EnvironmentData result{};

		result.asHandle = GetAccelerationStructure();
		result.rtPoints = m_RtPointsBuffer.DevicePointerCast<glm::vec3>();
		result.vwInfo = m_VoxelWorldInfo;
		result.edges = m_EdgeBuffer.DevicePointerCast<DiffractionEdge>();
		result.edgeCount = static_cast<uint32_t>(m_Edges.size());
		
		result.triangle.useFaceNormals = m_UseFaceNormals;
		result.triangle.indices = m_IndexBuffer.DevicePointerCast<uint32_t>();
		result.triangle.normals = m_NormalBuffer.DevicePointerCast<glm::vec3>();
		result.triangle.faces = m_FaceBuffer.DevicePointerCast<Face>();
		result.triangle.voxelToRtPointIndexMap = m_VoxelToRtPointIndexMapBuffer.DevicePointerCast<uint32_t>();

		result.ris.objectIds = m_RisData.objectIds.DevicePointerCast<uint32_t>();
		result.ris.normals = m_RisData.normals.DevicePointerCast<glm::vec3>();
		result.ris.cellWorldPositions = m_RisData.cellWorldPositions.DevicePointerCast<glm::vec3>();
		result.ris.cellObjectIds = m_RisData.cellObjectIds.DevicePointerCast<uint32_t>();
		result.ris.cellCount = m_RisData.cellCount;

		return result;
	}

	void TriangleMeshEnvironment::ComputeVisibility(const DeviceBuffer& params, const glm::uvec3& dims) const
	{
		KernelData::Get().GetStTrVisPipeline().LaunchAndSynchronize(params, dims);
	}

	void TriangleMeshEnvironment::DetermineLosPaths(const DeviceBuffer& params, const glm::uvec3& dims) const 
	{
		KernelData::Get().GetStTrTransmitLOSPipeline().LaunchAndSynchronize(params, dims);
	}

	void TriangleMeshEnvironment::Transmit(const DeviceBuffer& params, const glm::uvec3& dims) const 
	{
		KernelData::Get().GetStTrTransmitPipeline().LaunchAndSynchronize(params, dims);
	}

	void TriangleMeshEnvironment::Propagate(const DeviceBuffer& params, const glm::uvec3& dims) const 
	{
		KernelData::Get().GetStTrPropagatePipeline().LaunchAndSynchronize(params, dims);
	}

	void TriangleMeshEnvironment::RefineSpecular(const DeviceBuffer& params, const glm::uvec3& dims) const 
	{
		KernelData::Get().GetStTrRefineSpecularPipeline().LaunchAndSynchronize(params, dims);
	}

	void TriangleMeshEnvironment::RefineScatterer(const DeviceBuffer& params, const glm::uvec3& dims) const 
	{
		KernelData::Get().GetStTrRefineScattererPipeline().LaunchAndSynchronize(params, dims);
	}

	void TriangleMeshEnvironment::RefineDiffraction(const DeviceBuffer& params, const glm::uvec3& dims) const
	{
		KernelData::Get().GetStTrRefineDiffractionPipeline().LaunchAndSynchronize(params, dims);
	}

	void TriangleMeshEnvironment::ComputeRISPaths(const DeviceBuffer& params, const glm::uvec3& dims) const
	{
		KernelData::Get().GetStTrComputeRISPathsPipeline().LaunchAndSynchronize(params, dims);
	}

	bool TriangleMeshEnvironment::ComputeAabb(const glm::vec3* vertices, size_t numVertices)
	{
		m_Aabb.min = *vertices;
		m_Aabb.max = *vertices;
		for (size_t vertexIndex = 1; vertexIndex < numVertices; ++vertexIndex)
		{
			m_Aabb.min = glm::min(vertices[vertexIndex], m_Aabb.min);
			m_Aabb.max = glm::max(vertices[vertexIndex], m_Aabb.max);
		}

		constexpr float bias = 0.01f;
		m_Aabb.min -= bias;
		m_Aabb.max += bias;

		return numVertices > 0;
	}

	bool TriangleMeshEnvironment::ComputeVoxelWorld(float voxelSize)
	{
		glm::uvec3 voxelDimensions = glm::uvec3(glm::ceil((m_Aabb.max - m_Aabb.min) / voxelSize));
		m_VoxelWorldInfo = VoxelWorldInfo(m_Aabb.min, voxelSize, voxelDimensions);
		return voxelDimensions.x > 0 && voxelDimensions.y > 0 && voxelDimensions.z > 0 && voxelSize > 0.0f;
	}

	bool TriangleMeshEnvironment::ComputeRayReceptionPoints(const glm::vec3* vertices, const glm::uvec3* indices, const Face* faces, size_t numFaces)
	{
		TriangleData data{};
		DeviceBuffer triangleDataBuffer = DeviceBuffer(sizeof(TriangleData));
		DeviceBuffer rtPointCounterBuffer = DeviceBuffer(sizeof(uint32_t));
		rtPointCounterBuffer.MemsetZero();

		data.numFaces = static_cast<uint32_t>(numFaces);
		data.indices = m_IndexBuffer.DevicePointerCast<glm::uvec3>();
		data.vertices = m_VertexBuffer.DevicePointerCast<glm::vec3>();
		data.faces = m_FaceBuffer.DevicePointerCast<Face>();
		data.vwInfo = m_VoxelWorldInfo;

		data.rtPoints = m_RtPointsBuffer.DevicePointerCast<glm::vec3>();
		data.rtPointCounter = rtPointCounterBuffer.DevicePointerCast<uint32_t>();
		data.voxelToRtPointIndexMap = m_VoxelToRtPointIndexMapBuffer.DevicePointerCast<uint32_t>();
		triangleDataBuffer.Upload(&data, 1);

		CUdeviceptr ptr = triangleDataBuffer.GetRawHandle();
		void* kernelParams[] = { &ptr };
		constexpr uint32_t blockSize = 256u;
		uint32_t gridCount = Utils::GetLaunchCount(static_cast<uint32_t>(numFaces), blockSize);
		KernelData::Get().GetCreateTrianglePrimitivesKernel().LaunchAndSynchronize(glm::uvec3(gridCount, 1, 1), glm::uvec3(blockSize, 1, 1), 0u, kernelParams);

		rtPointCounterBuffer.Download(&m_RtPointCount, 1);
		return m_RtPointCount > 0;
	}

	bool TriangleMeshEnvironment::GenerateGpuData(const glm::vec3* vertices,
												  const glm::vec3* normals,
												  size_t numVertices,
												  const glm::uvec3* indices,
												  const Face* faces,
												  size_t numFaces)
	{
		m_VertexBuffer = DeviceBuffer::Create(vertices, numVertices);
		m_NormalBuffer = DeviceBuffer::Create(normals, numVertices);
		m_IndexBuffer = DeviceBuffer::Create(indices, numFaces);
		m_FaceBuffer = DeviceBuffer::Create(faces, numFaces);
		m_VoxelToRtPointIndexMapBuffer = DeviceBuffer(sizeof(uint32_t) * m_VoxelWorldInfo.dimensions.x * m_VoxelWorldInfo.dimensions.y * m_VoxelWorldInfo.dimensions.z);
		m_VoxelToRtPointIndexMapBuffer.Memset(Nimbus::Constants::InvalidPointIndex);
		m_RtPointsBuffer = DeviceBuffer(sizeof(glm::vec3) * m_VoxelWorldInfo.dimensions.x * m_VoxelWorldInfo.dimensions.y * m_VoxelWorldInfo.dimensions.z);

		m_AccelerationStructure = AccelerationStructure::CreateFromTriangles(m_VertexBuffer, static_cast<uint32_t>(numVertices), m_IndexBuffer, static_cast<uint32_t>(numFaces));
		return m_AccelerationStructure;
	}
}