#include "KernelData.hpp"
#include "Profiler.hpp"
#include <optix.h>
#include <array>

extern "C"
{
    unsigned char Ptx_Voxelization[];
    unsigned char Ptx_VCT[];
    unsigned char Ptx_GS[];
    unsigned char Ptx_GS_Voxelization[];
}

namespace VCT
{
    bool KernelData::Initialize()
    {
        if (!s_KernelData)
            s_KernelData = std::unique_ptr<KernelData>(new KernelData());

        return DeviceContext::Get()
            && s_KernelData->m_VoxelizationModule
            && s_KernelData->m_VoxelizePointCloudKernel
            && s_KernelData->m_FillTextureDataKernel
            && s_KernelData->m_WriteRefineAabbKernel
            && s_KernelData->m_WriteRefinePrimitiveNeighborsKernel
            && s_KernelData->m_VoxelizationConstantBuffer
            && s_KernelData->m_RtModule
            && s_KernelData->m_TransmitPipeline
            && s_KernelData->m_PropagationPipeline
            && s_KernelData->m_RefinePipeline;
    }

    void KernelData::Destroy()
    {
        s_KernelData.reset(nullptr);
    }

    KernelData::KernelData()
    {
        PROFILE_SCOPE();
        DeviceContext::Get();
        m_VoxelizationModule = Module(reinterpret_cast<const char*>(Ptx_Voxelization));
        m_VoxelizePointCloudKernel = m_VoxelizationModule.LoadKernel("VoxelizePointCloud");
        m_FillTextureDataKernel = m_VoxelizationModule.LoadKernel("FillTextureData");
        m_WriteRefineAabbKernel = m_VoxelizationModule.LoadKernel("WriteRefineAabb");
        m_WriteRefinePrimitiveNeighborsKernel = m_VoxelizationModule.LoadKernel("WriteRefinePrimitiveNeighbors");
        m_VoxelizationConstantBuffer = m_VoxelizationModule.LoadConstantBuffer("data");
        
        OptixModuleCompileOptions moduleCompileOptions{};
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        OptixPipelineCompileOptions pipelineCompileOptions{};
        pipelineCompileOptions.numPayloadValues = 7;
        pipelineCompileOptions.numAttributeValues = 4;
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "data";

        OptixPipelineLinkOptions pipelineLinkOptions{};
        pipelineLinkOptions.maxTraceDepth = 1;
        m_RtModule = RTModule(reinterpret_cast<const char*>(Ptx_VCT), pipelineCompileOptions, moduleCompileOptions);

        std::array<OptixProgramGroupDesc, 3> pgDescs{};
        pgDescs[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDescs[0].raygen.module = m_RtModule.GetRawHandle();
        pgDescs[0].raygen.entryFunctionName = "__raygen__VCT";

        pgDescs[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDescs[1].miss.module = m_RtModule.GetRawHandle();
        pgDescs[1].miss.entryFunctionName = "__miss__VCT";

        pgDescs[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDescs[2].hitgroup.moduleCH = m_RtModule.GetRawHandle();
        pgDescs[2].hitgroup.entryFunctionNameCH = "__closesthit__VCT";
        pgDescs[2].hitgroup.moduleIS = m_RtModule.GetRawHandle();
        pgDescs[2].hitgroup.entryFunctionNameIS = "__intersection__VCT";


        pgDescs[0].raygen.entryFunctionName = "__raygen__TransmitVCT";
        m_TransmitPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[0].raygen.entryFunctionName = "__raygen__VCT";
        m_PropagationPipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);

        pgDescs[0].raygen.entryFunctionName = "__raygen__Refine";
        pgDescs[1].miss.entryFunctionName = "__miss__Refine";
        pgDescs[2].hitgroup.entryFunctionNameIS = "__intersection__Refine";
        pgDescs[2].hitgroup.entryFunctionNameCH = "__closesthit__Refine";
        pgDescs[2].hitgroup.entryFunctionNameAH = nullptr;
        pgDescs[2].hitgroup.moduleAH = nullptr;
        m_RefinePipeline = RTPipeline(pgDescs.data(), pgDescs.size(), pipelineCompileOptions, pipelineLinkOptions);
    }
}