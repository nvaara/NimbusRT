#pragma once
#include "Kernel.hpp"

namespace VCT
{
	class KernelData
	{
	public:
		static bool Initialize();
		static void Destroy();
		static const KernelData& Get() { return *s_KernelData; }

		KernelData(const KernelData&) = delete;
		KernelData(KernelData&&) = delete;
		KernelData& operator=(const KernelData&) = delete;
		KernelData& operator=(KernelData&&) = delete;

		const Kernel& GetVoxelizePointCloudKernel() const { return m_VoxelizePointCloudKernel; }
		const Kernel& GetFillTextureDataKernel() const { return m_FillTextureDataKernel;}
		const Kernel& GetWriteRefineAabbKernel() const { return m_WriteRefineAabbKernel; }
		const Kernel& GetWriteRefinePrimitiveNeighborsKernel() const { return m_WriteRefinePrimitiveNeighborsKernel; }
		const DeviceBuffer& GetVoxelizationConstantBuffer() const { return m_VoxelizationConstantBuffer; }

		const RTPipeline& GetTransmitPipeline() const { return m_TransmitPipeline; }
		const RTPipeline& GetPropagationPipeline() const { return m_PropagationPipeline; }
		const RTPipeline& GetRefinePipeline() const { return m_RefinePipeline; }

	private:
		KernelData();

	private:
		inline static std::unique_ptr<KernelData> s_KernelData = nullptr;

		Module m_VoxelizationModule;
		Kernel m_VoxelizePointCloudKernel;
		Kernel m_FillTextureDataKernel;
		Kernel m_WriteRefineAabbKernel;
		Kernel m_WriteRefinePrimitiveNeighborsKernel;
		DeviceBuffer m_VoxelizationConstantBuffer;

		RTModule m_RtModule;
		RTPipeline m_TransmitPipeline;
		RTPipeline m_PropagationPipeline;
		RTPipeline m_RefinePipeline;
	};
}