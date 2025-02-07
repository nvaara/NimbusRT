#include "CudaUtils.hpp"
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <array>
#include "Logger.hpp"

namespace Nimbus
{
	namespace
	{
		struct SbtData {};
		struct SbtRecord
		{
			__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
			SbtData data = {};
		};
	}

	DeviceContext& DeviceContext::Get()
	{
		static DeviceContext context = DeviceContext();
		return context;
	}

	void DeviceContext::Synchronize()
	{
		CU_CHECK(cuCtxSynchronize());
	}

	void DeviceContext::StreamSynchronize(CUstream stream)
	{
		CU_CHECK(cuStreamSynchronize(stream));
	}

	DeviceContext::DeviceContext()
		: m_CudaDevice(0)
		, m_CudaContext(nullptr)
		, m_OptixContext(nullptr)
	{
		CU_CHECK(cuInit(0));
		CU_CHECK(cuDeviceGet(&m_CudaDevice, 0));
		CU_CHECK(cuCtxCreate(&m_CudaContext, 0, m_CudaDevice));
		OPTIX_CHECK(optixInit());
		OPTIX_CHECK(optixDeviceContextCreate(m_CudaContext, nullptr, &m_OptixContext));
	}

	Module::Module(const std::string_view& ptxData)
		: m_Module(nullptr)
	{
		if (!ptxData.empty())
			CU_CHECK(cuModuleLoadDataEx(&m_Module, ptxData.data(), 0, nullptr, nullptr));
	}

	Kernel Module::LoadKernel(const std::string_view& functionName)
	{
		CUfunction result = nullptr;
		CU_CHECK(cuModuleGetFunction(&result, m_Module, functionName.data()));
		return Kernel(result);
	}

	DeviceBuffer Module::LoadConstantBuffer(const std::string_view& constantName)
	{
		CUdeviceptr ptr = NULL;
		size_t size = 0;
		CU_CHECK(cuModuleGetGlobal(&ptr, &size, m_Module, constantName.data()));
		return DeviceBuffer(ptr, size);
	}

	RTModule::RTModule(const std::string_view& ptxData, const OptixPipelineCompileOptions& pipelineCompileOptions, const OptixModuleCompileOptions& moduleCompileOptions)
		: m_Module(nullptr)
	{
		if (!ptxData.empty())
			OPTIX_CHECK(optixModuleCreate(DeviceContext::Get().GetOptixContext(), &moduleCompileOptions, &pipelineCompileOptions, ptxData.data(), ptxData.size(), nullptr, nullptr, &m_Module));
	}
	
	RTPipeline::RTPipeline(const OptixProgramGroupDesc* pgDescs,
						   size_t numDescs,
						   const OptixPipelineCompileOptions& compileOptions,
						   const OptixPipelineLinkOptions& linkOptions,
						   const OptixProgramGroupOptions& options)
		: m_Pipeline(nullptr)
		, m_SbtTable({})
	{
		std::array<char, 2048> log{};
		size_t logSize = log.size();
		m_ProgramGroups.resize(numDescs);

		OPTIX_CHECK_LOG(optixProgramGroupCreate(DeviceContext::Get().GetOptixContext(),
			pgDescs,
			static_cast<uint32_t>(numDescs),
			&options,
			log.data(), &logSize,
			m_ProgramGroups.data()), log.data());

		OPTIX_CHECK_LOG(optixPipelineCreate(DeviceContext::Get().GetOptixContext(),
			&compileOptions,
			&linkOptions,
			m_ProgramGroups.data(), static_cast<uint32_t>(m_ProgramGroups.size()),
			log.data(), &logSize,
			&m_Pipeline), log.data());

		m_SbtBuffer = DeviceBuffer(sizeof(SbtRecord) * m_ProgramGroups.size());
		std::vector<SbtRecord> sbtRecords;
		sbtRecords.resize(m_ProgramGroups.size());

		for (size_t i = 0; i < m_ProgramGroups.size(); ++i)
		{
			CUdeviceptr ptr = m_SbtBuffer.GetRawHandle() + sizeof(SbtRecord) * i;
			switch (pgDescs[i].kind)
			{
			case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
				m_SbtTable.raygenRecord = ptr;
				break;
			case OPTIX_PROGRAM_GROUP_KIND_MISS:
			{
				if (!m_SbtTable.missRecordBase)
				{
					m_SbtTable.missRecordBase = ptr;
					m_SbtTable.missRecordStrideInBytes = sizeof(SbtRecord);
				}
				m_SbtTable.missRecordCount++;
				break;
			}
			case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
				break;
			case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
			{
				if (!m_SbtTable.hitgroupRecordBase)
				{
					m_SbtTable.hitgroupRecordBase = ptr;
					m_SbtTable.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
				}
				m_SbtTable.hitgroupRecordCount++;
				break;
			}
			case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
				break;
			}
			OPTIX_CHECK(optixSbtRecordPackHeader(m_ProgramGroups[i], &sbtRecords[i]));
		}
		m_SbtBuffer.Upload(sbtRecords.data(), sbtRecords.size());
	}
	
	AccelerationStructure AccelerationStructure::CreateFromAabbs(const DeviceBuffer& aabbBuffer, uint32_t primitiveCount)
	{
		if (primitiveCount == 0)
		{
			LOG("Attempted to create acceleration structure with 0 primitives.");
			return AccelerationStructure();
		}
		CUdeviceptr p = aabbBuffer.GetRawHandle();

		OptixBuildInput buildInput{};
		buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		buildInput.customPrimitiveArray.aabbBuffers = &p;
		buildInput.customPrimitiveArray.numPrimitives = primitiveCount;

		uint32_t inputFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
		buildInput.customPrimitiveArray.flags = &inputFlags;
		buildInput.customPrimitiveArray.numSbtRecords = 1;
		return AccelerationStructure(buildInput);
	}

	AccelerationStructure AccelerationStructure::CreateFromTriangles(const Nimbus::DeviceBuffer& vertexBuffer, uint32_t numVertices, const Nimbus::DeviceBuffer& indexBuffer, uint32_t numFaces)
	{
		OptixBuildInput buildInput{};
		buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		CUdeviceptr v = vertexBuffer.GetRawHandle();
		buildInput.triangleArray.vertexBuffers = &v;
		buildInput.triangleArray.numVertices = numVertices;
		buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		buildInput.triangleArray.indexBuffer = indexBuffer.GetRawHandle();
		buildInput.triangleArray.numIndexTriplets = numFaces;
		buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

		uint32_t inputFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
		buildInput.triangleArray.flags = &inputFlags;
		buildInput.triangleArray.numSbtRecords = 1;

		return AccelerationStructure(buildInput);
	}

	AccelerationStructure AccelerationStructure::CreateFromInstances(const Nimbus::DeviceBuffer& instanceBuffer, uint32_t numInstances)
	{
		OptixBuildInput buildInput{};
		buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		buildInput.instanceArray.instances = instanceBuffer.GetRawHandle();
		buildInput.instanceArray.instanceStride = 0u;
		buildInput.instanceArray.numInstances = numInstances;
		return AccelerationStructure(buildInput);
	}

	AccelerationStructure::AccelerationStructure(const OptixBuildInput& buildInput)
		: m_AsHandle(0)
	{
		OptixAccelBuildOptions buildOptions{};
		buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes blasBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(DeviceContext::Get().GetOptixContext(), &buildOptions, &buildInput, 1, &blasBufferSizes));

		DeviceBuffer compactedSizeBuffer = DeviceBuffer(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc{};
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.GetRawHandle();

		DeviceBuffer tempBuffer = DeviceBuffer(blasBufferSizes.tempSizeInBytes);
		DeviceBuffer outputBuffer = DeviceBuffer(blasBufferSizes.outputSizeInBytes);

		OPTIX_CHECK(optixAccelBuild(DeviceContext::Get().GetOptixContext(),
			0,
			&buildOptions,
			&buildInput,
			1,
			tempBuffer.GetRawHandle(),
			tempBuffer.GetSize(),
			outputBuffer.GetRawHandle(),
			outputBuffer.GetSize(),
			&m_AsHandle,
			&emitDesc,
			1
		));

		CU_CHECK(cuStreamSynchronize(0));

		uint64_t compactedSize;
		compactedSizeBuffer.Download(&compactedSize, 1);

		m_AsBuffer = DeviceBuffer(compactedSize);

		OPTIX_CHECK(optixAccelCompact(DeviceContext::Get().GetOptixContext(),
			0,
			m_AsHandle,
			m_AsBuffer.GetRawHandle(),
			m_AsBuffer.GetSize(),
			&m_AsHandle));

		CU_CHECK(cuStreamSynchronize(0));
	}
	
	void Kernel::Launch(uint32_t gridX,
		uint32_t gridY,
		uint32_t gridZ,
		uint32_t blockX,
		uint32_t blockY,
		uint32_t blockZ,
		CUstream stream,
		void** kernelParams,
		unsigned int sharedMemBytes,
		void** extra) const
	{
		CU_CHECK(cuLaunchKernel(m_Function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes, stream, kernelParams, extra));
	}

	void Kernel::Launch(const glm::uvec3& grid,
					    const glm::uvec3& block,
					    CUstream stream,
					    void** kernelParams,
					    unsigned int sharedMemBytes,
					    void** extra) const
	{
		return Launch(grid.x, grid.y, grid.z, block.x, block.y, block.z, stream, kernelParams, sharedMemBytes, extra);
	}

	void Kernel::LaunchAndSynchronize(uint32_t gridX,
									  uint32_t gridY,
									  uint32_t gridZ,
									  uint32_t blockX,
									  uint32_t blockY,
									  uint32_t blockZ,
									  CUstream stream,
									  void** kernelParams,
									  unsigned int sharedMemBytes,
									  void** extra) const
	{
		Launch(gridX, gridY, gridZ, blockX, blockY, blockZ, stream, kernelParams, sharedMemBytes, extra);
		CUDA_CHECK(cudaStreamSynchronize(stream));
	}

	void Kernel::LaunchAndSynchronize(const glm::uvec3& grid,
									  const glm::uvec3& block,
									  CUstream stream,
									  void** kernelParams,
									  unsigned int sharedMemBytes,
									  void** extra) const
	{
		return LaunchAndSynchronize(grid.x, grid.y, grid.z, block.x, block.y, block.z, stream, kernelParams, sharedMemBytes, extra);
	}

	void RTPipeline::Launch(const DeviceBuffer& pipelineParams, uint32_t width, uint32_t height, uint32_t depth, CUstream stream) const
	{
		OPTIX_CHECK(optixLaunch(m_Pipeline, stream, pipelineParams.GetRawHandle(), pipelineParams.GetSize(), &m_SbtTable, width, height, depth));
	}

	void RTPipeline::Launch(const DeviceBuffer& pipelineParams, const glm::uvec3& dimensions, CUstream stream) const
	{
		return Launch(pipelineParams, dimensions.x, dimensions.y, dimensions.z, stream);
	}

	void RTPipeline::LaunchAndSynchronize(const DeviceBuffer& pipelineParams, uint32_t width, uint32_t height, uint32_t depth, CUstream stream) const
	{
		Launch(pipelineParams, width, height, depth, stream);
		CUDA_CHECK(cudaStreamSynchronize(stream));
	}

	void RTPipeline::LaunchAndSynchronize(const DeviceBuffer& pipelineParams, const glm::uvec3& dimensions, CUstream stream) const
	{
		LaunchAndSynchronize(pipelineParams, dimensions.x, dimensions.y, dimensions.z, stream);
	}
}