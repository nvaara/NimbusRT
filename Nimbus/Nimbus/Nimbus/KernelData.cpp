#include "KernelData.hpp"
#include "Profiler.hpp"
#include <optix.h>
#include <array>
#include <memory>

extern "C"
{
    extern unsigned char Ptx_Primitive[];
    extern unsigned char Ptx_Triangle[];
    extern unsigned char Ptx_Trace[];
    extern unsigned char Ptx_Visibility[];
    extern unsigned char Ptx_Refine[];
    extern unsigned char Ptx_Coverage[];
}


namespace Nimbus
{
    namespace
    {
        std::unique_ptr<KernelData> s_KernelData = nullptr;

        void SetModule(RTModule& module, std::array<OptixProgramGroupDesc, 4>& pgDescs)
        {
            for (OptixProgramGroupDesc& desc : pgDescs)
            {
                switch (desc.kind)
                { 
                case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
                {
                    desc.raygen.module = module.GetRawHandle();
                    break;
                }
                case OPTIX_PROGRAM_GROUP_KIND_MISS:
                {
                    desc.miss.module = module.GetRawHandle();
                    break;
                }
                case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
                {
                    if (desc.hitgroup.entryFunctionNameCH)
                        desc.hitgroup.moduleCH = module.GetRawHandle();

                    if (desc.hitgroup.entryFunctionNameAH)
                        desc.hitgroup.moduleAH = module.GetRawHandle();

                    if (desc.hitgroup.entryFunctionNameIS)
                        desc.hitgroup.moduleIS = module.GetRawHandle();
                    break;
                }
                }
            }
        }
    }

    bool KernelData::Initialize()
    {
        if (!s_KernelData)
            s_KernelData = std::unique_ptr<KernelData>(new KernelData());

        return DeviceContext::Get()

            && s_KernelData->m_StModule
            && s_KernelData->m_StCreatePrimitivesKernel
            && s_KernelData->m_StConstantBuffer
            
            && s_KernelData->m_StCoverageModule
            && s_KernelData->m_StCoveragePointsKernel
            && s_KernelData->m_StCoverageConstantBuffer

            && s_KernelData->m_StTriangleModule
            && s_KernelData->m_StCreateTrianglePrimitivesKernel

            && s_KernelData->m_StRTModule
            && s_KernelData->m_StVisModule
            && s_KernelData->m_StRefineModule

            && s_KernelData->m_StVisPipeline
            && s_KernelData->m_StTransmitPipeline
            && s_KernelData->m_StTransmitLOSPipeline
            && s_KernelData->m_StPropagatePipeline
            && s_KernelData->m_StRefineSpecularPipeline
            && s_KernelData->m_StRefineScattererPipeline
            && s_KernelData->m_StRefineDiffractionPipeline
            && s_KernelData->m_StComputeRISPathsPipeline

            && s_KernelData->m_StTrVisPipeline
            && s_KernelData->m_StTrTransmitPipeline
            && s_KernelData->m_StTrTransmitLOSPipeline
            && s_KernelData->m_StTrPropagatePipeline
            && s_KernelData->m_StTrRefineSpecularPipeline
            && s_KernelData->m_StTrRefineScattererPipeline
            && s_KernelData->m_StTrRefineDiffractionPipeline
            && s_KernelData->m_StTrComputeRISPathsPipeline;
    }

    void KernelData::Destroy()
    {
        s_KernelData.reset(nullptr);
    }

    const KernelData& KernelData::Get()
    {
        return *s_KernelData;
    }

    KernelData::KernelData()
    {
        PROFILE_SCOPE();
        DeviceContext::Get();

        m_StModule = Module(reinterpret_cast<const char*>(Ptx_Primitive));
        m_StCreatePrimitivesKernel = m_StModule.LoadKernel("CreatePrimitives");
        m_StConstantBuffer = m_StModule.LoadConstantBuffer("data");

        m_StCoverageModule = Module(reinterpret_cast<const char*>(Ptx_Coverage));
        m_StCoveragePointsKernel = m_StCoverageModule.LoadKernel("CoveragePoints");
        m_StCoverageConstantBuffer = m_StCoverageModule.LoadConstantBuffer("data");

        m_StTriangleModule = Module(reinterpret_cast<const char*>(Ptx_Triangle));
        m_StCreateTrianglePrimitivesKernel = m_StTriangleModule.LoadKernel("CreateTrianglePrimitives");

        OptixModuleCompileOptions moduleCompileOptions{};
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        OptixPipelineCompileOptions pipelineCompileOptions{};
        pipelineCompileOptions.numPayloadValues = 2;
        pipelineCompileOptions.numAttributeValues = 4;
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "data";

        OptixPipelineLinkOptions pipelineLinkOptions{};
        pipelineLinkOptions.maxTraceDepth = 1;
        std::array<OptixProgramGroupDesc, 4> pgDescs{};
        m_StRTModule = RTModule(reinterpret_cast<const char*>(Ptx_Trace), pipelineCompileOptions, moduleCompileOptions);
        m_StRefineModule = RTModule(reinterpret_cast<const char*>(Ptx_Refine), pipelineCompileOptions, moduleCompileOptions);
        m_StVisModule = RTModule(reinterpret_cast<const char*>(Ptx_Visibility), pipelineCompileOptions, moduleCompileOptions);
        
        pgDescs[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDescs[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDescs[1].miss.entryFunctionName = "__miss__ST";

        OptixProgramGroupDesc pointCloudDesc{};
        pointCloudDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pointCloudDesc.hitgroup.entryFunctionNameCH = "__closesthit__ST";
        pointCloudDesc.hitgroup.entryFunctionNameIS = "__intersection__ST";
        
        OptixProgramGroupDesc risDesc{};
        risDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        risDesc.hitgroup.entryFunctionNameCH = "__closesthit__ST_RIS";
        pgDescs[3] = risDesc;

        OptixProgramGroupDesc triangleDesc{};
        triangleDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        triangleDesc.hitgroup.entryFunctionNameCH = "__closesthit__ST_TR";

        pgDescs[0].raygen.entryFunctionName = "__raygen__Visibility";
        pgDescs[2] = pointCloudDesc;
        SetModule(m_StVisModule, pgDescs);
        m_StVisPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);
        
        pgDescs[2] = triangleDesc;
        SetModule(m_StVisModule, pgDescs);
        m_StTrVisPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);
        
        pgDescs[0].raygen.entryFunctionName = "__raygen__Transmit";
        pgDescs[2] = pointCloudDesc;
        SetModule(m_StRTModule, pgDescs);
        m_StTransmitPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[2] = triangleDesc;
        SetModule(m_StRTModule, pgDescs);
        m_StTrTransmitPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[0].raygen.entryFunctionName = "__raygen__TransmitLOS";
        pgDescs[2] = pointCloudDesc;
        SetModule(m_StRTModule, pgDescs);
        m_StTransmitLOSPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[2] = triangleDesc;
        SetModule(m_StRTModule, pgDescs);
        m_StTrTransmitLOSPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[0].raygen.entryFunctionName = "__raygen__Propagate";
        pgDescs[2] = pointCloudDesc;
        SetModule(m_StRTModule, pgDescs);
        m_StPropagatePipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[2] = triangleDesc;
        SetModule(m_StRTModule, pgDescs);
        m_StTrPropagatePipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[0].raygen.entryFunctionName = "__raygen__RefineSpecular";
        pgDescs[2] = pointCloudDesc;
        SetModule(m_StRefineModule, pgDescs);
        m_StRefineSpecularPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[2] = triangleDesc;
        SetModule(m_StRefineModule, pgDescs);
        m_StTrRefineSpecularPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[0].raygen.entryFunctionName = "__raygen__RefineScatterer";
        pgDescs[2] = pointCloudDesc;
        SetModule(m_StRefineModule, pgDescs);
        m_StRefineScattererPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[2] = triangleDesc;
        SetModule(m_StRefineModule, pgDescs);
        m_StTrRefineScattererPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[0].raygen.entryFunctionName = "__raygen__RefineDiffraction";
        pgDescs[2] = pointCloudDesc;
        SetModule(m_StRefineModule, pgDescs);
        m_StRefineDiffractionPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[2] = triangleDesc;
        SetModule(m_StRefineModule, pgDescs);
        m_StTrRefineDiffractionPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[0].raygen.entryFunctionName = "__raygen__ComputeRisPaths";
        pgDescs[2] = pointCloudDesc;
        SetModule(m_StRefineModule, pgDescs);
        m_StComputeRISPathsPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[2] = triangleDesc;
        SetModule(m_StRefineModule, pgDescs);
        m_StTrComputeRISPathsPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);
    }
}