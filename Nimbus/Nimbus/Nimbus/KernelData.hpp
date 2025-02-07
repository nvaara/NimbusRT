#pragma once
#include "CudaUtils.hpp"
#include <memory>

namespace Nimbus
{
	class KernelData
	{
	public:
		static bool Initialize();
		static void Destroy();
		static const KernelData& Get();

		KernelData(const KernelData&) = delete;
		KernelData(KernelData&&) = delete;
		KernelData& operator=(const KernelData&) = delete;
		KernelData& operator=(KernelData&&) = delete;

		const Kernel& GetStCreatePrimitivesKernel() const { return m_StCreatePrimitivesKernel; }
		const DeviceBuffer& GetStConstantBuffer() const { return m_StConstantBuffer; }
		const Kernel& GetStCoveragePointsKernel() const { return m_StCoveragePointsKernel; }
		const DeviceBuffer& GetStCoverageConstantBuffer() const { return m_StCoverageConstantBuffer; }
		const Kernel& GetCreateTrianglePrimitivesKernel() const { return m_StCreateTrianglePrimitivesKernel; }

		const RTPipeline& GetStVisPipeline() const { return m_StVisPipeline; }
		const RTPipeline& GetStTransmitPipeline() const { return m_StTransmitPipeline; }
		const RTPipeline& GetStTransmitLOSPipeline() const { return m_StTransmitLOSPipeline; }
		const RTPipeline& GetStPropagatePipeline() const { return m_StPropagatePipeline; }
		const RTPipeline& GetStRefineSpecularPipeline() const { return m_StRefineSpecularPipeline; }
		const RTPipeline& GetStRefineScattererPipeline() const { return m_StRefineScattererPipeline; }
		const RTPipeline& GetStRefineDiffractionPipeline() const { return m_StRefineDiffractionPipeline; }
		const RTPipeline& GetStComputeRISPathsPipeline() const { return m_StComputeRISPathsPipeline; }

		const RTPipeline& GetStTrVisPipeline() const { return m_StTrVisPipeline; }
		const RTPipeline& GetStTrTransmitPipeline() const { return m_StTrTransmitPipeline; }
		const RTPipeline& GetStTrTransmitLOSPipeline() const { return m_StTrTransmitLOSPipeline; }
		const RTPipeline& GetStTrPropagatePipeline() const { return m_StTrPropagatePipeline; }
		const RTPipeline& GetStTrRefineSpecularPipeline() const { return m_StTrRefineSpecularPipeline; }
		const RTPipeline& GetStTrRefineScattererPipeline() const { return m_StTrRefineScattererPipeline; }
		const RTPipeline& GetStTrRefineDiffractionPipeline() const { return m_StTrRefineDiffractionPipeline; }
		const RTPipeline& GetStTrComputeRISPathsPipeline() const { return m_StTrComputeRISPathsPipeline; }

	private:
		KernelData();

	private:
		Module m_StModule;
		Kernel m_StCreatePrimitivesKernel;
		DeviceBuffer m_StConstantBuffer;
		Module m_StCoverageModule;
		Kernel m_StCoveragePointsKernel;
		DeviceBuffer m_StCoverageConstantBuffer;
		Module m_StTriangleModule;
		Kernel m_StCreateTrianglePrimitivesKernel;

		RTModule m_StRTModule;
		RTModule m_StVisModule;
		RTModule m_StRefineModule;
		RTPipeline m_StVisPipeline;
		RTPipeline m_StTransmitPipeline;
		RTPipeline m_StTransmitLOSPipeline;
		RTPipeline m_StPropagatePipeline;
		RTPipeline m_StRefineSpecularPipeline;
		RTPipeline m_StRefineScattererPipeline;
		RTPipeline m_StRefineDiffractionPipeline;
		RTPipeline m_StComputeRISPathsPipeline;

		RTPipeline m_StTrVisPipeline;
		RTPipeline m_StTrTransmitPipeline;
		RTPipeline m_StTrTransmitLOSPipeline;
		RTPipeline m_StTrPropagatePipeline;
		RTPipeline m_StTrRefineSpecularPipeline;
		RTPipeline m_StTrRefineScattererPipeline;
		RTPipeline m_StTrRefineDiffractionPipeline;
		RTPipeline m_StTrComputeRISPathsPipeline;
	};
}