#pragma once
#include <string_view>
#include <vector>
#include "DeviceBuffer.hpp"
#include <glm/glm.hpp>

namespace Nimbus
{
	class DeviceContext
	{
	public:
		static DeviceContext& Get();
		void Synchronize();
		void StreamSynchronize(CUstream stream = 0);

		operator bool() const { return m_CudaContext != nullptr && m_OptixContext != nullptr; }
		
		CUdevice GetCudaDevice() const { return m_CudaDevice; }
		CUcontext GetCudaContext() const { return m_CudaContext; }
		OptixDeviceContext GetOptixContext() const { return m_OptixContext; }

		DeviceContext(const DeviceContext&) = delete;
		DeviceContext(DeviceContext&&) = delete;
		DeviceContext& operator=(const DeviceContext&) = delete;
		DeviceContext& operator=(DeviceContext&&) = delete;

	private:
		DeviceContext();

	private:
		CUdevice m_CudaDevice;
		CUcontext m_CudaContext;
		OptixDeviceContext m_OptixContext;
	};
	
	class Kernel
	{
	public:
		Kernel() : m_Function(nullptr) {}
		Kernel(CUfunction func) : m_Function(func) {}

		operator bool() const { return m_Function != nullptr; }
		CUfunction GetRawHandle() const { return m_Function; }
		
		void Launch(uint32_t gridX,
					uint32_t gridY,
					uint32_t gridZ,
					uint32_t blockX,
					uint32_t blockY,
					uint32_t blockZ,
					CUstream stream = 0,
					void** kernelParams = nullptr,
					unsigned int sharedMemBytes = 0,
					void** extra = nullptr) const;

		void Launch(const glm::uvec3& grid,
					const glm::uvec3& block,
					CUstream stream = 0,
					void** kernelParams = nullptr,
					unsigned int sharedMemBytes = 0,
					void** extra = nullptr) const;

		void LaunchAndSynchronize(uint32_t gridX,
								  uint32_t gridY,
								  uint32_t gridZ,
								  uint32_t blockX,
								  uint32_t blockY,
								  uint32_t blockZ,
								  CUstream stream = 0,
								  void** kernelParams = nullptr,
								  unsigned int sharedMemBytes = 0,
								  void** extra = nullptr) const;

		void LaunchAndSynchronize(const glm::uvec3& grid,
								  const glm::uvec3& block,
								  CUstream stream = 0,
								  void** kernelParams = nullptr,
								  unsigned int sharedMemBytes = 0,
								  void** extra = nullptr) const;

	private:
		CUfunction m_Function;
	};

	class Module
	{
	public:
		Module() : m_Module(nullptr) {}
		Module(const std::string_view& ptxData);
		operator bool() const { return m_Module != nullptr; }
		CUmodule GetRawHandle() const { return m_Module; }

		Kernel LoadKernel(const std::string_view& functionName);
		DeviceBuffer LoadConstantBuffer(const std::string_view& constantName);

	private:
		CUmodule m_Module;
	};

	class RTModule
	{
	public:
		RTModule(): m_Module(nullptr) {}
		RTModule(const std::string_view& ptxData, const OptixPipelineCompileOptions& pipelineCompileOptions, const OptixModuleCompileOptions& moduleCompileOptions);
		
		operator bool() const { return m_Module != nullptr; }
		OptixModule GetRawHandle() const { return m_Module; }

	private:
		OptixModule m_Module;
	};

	class RTPipeline
	{
	public:
		RTPipeline() : m_Pipeline(nullptr), m_SbtTable({}) {}
		RTPipeline(const OptixProgramGroupDesc* pgDescs,
				   size_t numDescs,
				   const OptixPipelineCompileOptions& compileOptions,
				   const OptixPipelineLinkOptions& linkOptions,
				   const OptixProgramGroupOptions& options = {});

		operator bool() const { return m_Pipeline != nullptr; }
		void Launch(const DeviceBuffer& pipelineParams, uint32_t width, uint32_t height, uint32_t depth, CUstream stream = 0) const;
		void Launch(const DeviceBuffer& pipelineParams, const glm::uvec3& dimensions, CUstream stream = 0) const;
		void LaunchAndSynchronize(const DeviceBuffer& pipelineParams, uint32_t width, uint32_t height, uint32_t depth, CUstream stream = 0) const;
		void LaunchAndSynchronize(const DeviceBuffer& pipelineParams, const glm::uvec3& dimensions, CUstream stream = 0) const;

	private:
		OptixPipeline m_Pipeline;
		std::vector<OptixProgramGroup> m_ProgramGroups;
		DeviceBuffer m_SbtBuffer;
		OptixShaderBindingTable m_SbtTable;
	};

	class AccelerationStructure
	{
	public:
		static AccelerationStructure CreateFromAabbs(const Nimbus::DeviceBuffer& aabbBuffer, uint32_t primitiveCount);
		static AccelerationStructure CreateFromTriangles(const Nimbus::DeviceBuffer& vertexBuffer, uint32_t numVertices, const Nimbus::DeviceBuffer& indexBuffer, uint32_t numFaces);
		static AccelerationStructure CreateFromInstances(const Nimbus::DeviceBuffer& instanceBuffer, uint32_t numInstances);

		AccelerationStructure() : m_AsHandle(0) {}
		AccelerationStructure(const OptixBuildInput& buildInput);

		operator bool() const { return m_AsHandle != 0; }
		bool IsValid() const { return *this; }

		OptixTraversableHandle GetRawHandle() const { return m_AsHandle; }

	private:
		OptixTraversableHandle m_AsHandle;
		DeviceBuffer m_AsBuffer;
	};
}