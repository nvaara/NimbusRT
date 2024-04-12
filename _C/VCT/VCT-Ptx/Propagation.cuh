#pragma once

#ifndef __CUDACC__
	#define __CUDACC__
#endif

#include "Types.hpp"
#include "Utils.hpp"
#include "TextureTraverser.cuh"

inline __device__ bool SkipIntersectIE(const VCT::PropagationData& parent, VCT::IEType type)
{
	return parent.tpData.traceData.numInteractions <= 2 && type == VCT::IEType::Receiver;
}

inline __device__ bool IsValidInteraction(VCT::IEType ieType, const VCT::TraceProcessingData& tpData, const VCT::ConeTracingData& coneTracingData)
{
	bool spaceForInteraction = tpData.traceData.numInteractions < coneTracingData.maximumNumberOfInteractions;
	bool validInteraction = ieType != VCT::IEType::Edge || tpData.numDiffractions < coneTracingData.maximumNumberOfDiffractions;
	bool isReceiver = ieType == VCT::IEType::Receiver;
	return (isReceiver) || (spaceForInteraction && validInteraction);
}

template <uint32_t Size>
inline __device__ void InitVoxelHistory(uint32_t history[Size][Size][Size])
{
	#pragma unroll
	for (uint32_t x = 0; x < Size; ++x)
		#pragma unroll
		for (uint32_t y = 0; y < Size; ++y)
			#pragma unroll
			for (uint32_t z = 0; z < Size; ++z)
				history[x][y][z] = VCT::Constants::InvalidPointIndex;
}

template <uint32_t Size>
inline __device__ void CopyVoxelHistory(uint32_t dst[Size][Size][Size], uint32_t src[Size][Size][Size])
{
	#pragma unroll
	for (uint32_t x = 0; x < Size; ++x)
		#pragma unroll
		for (uint32_t y = 0; y < Size; ++y)
			#pragma unroll
			for (uint32_t z = 0; z < Size; ++z)
				dst[x][y][z] = src[x][y][z];
}

template <uint32_t N>
inline __device__ void KernelFetchVoxelIDs(const glm::vec3& currentVoxelCenter, const VCT::ConeTracingData& coneTracingData, uint32_t voxels[N][N][N])
{
	constexpr int32_t Range = (N - 1) / 2;
	static_assert(Range > 0, "Range out of scope.");

#pragma unroll
	for (int32_t x = -Range; x <= Range; ++x)
#pragma unroll
		for (int32_t y = -Range; y <= Range; ++y)
#pragma unroll
			for (int32_t z = -Range; z <= Range; ++z)
				voxels[x + Range][y + Range][z + Range] = tex3D<uint2>(coneTracingData.voxelTexture, currentVoxelCenter.x + static_cast<float>(x),
					currentVoxelCenter.y + static_cast<float>(y),
					currentVoxelCenter.z + static_cast<float>(z)).x;
}

template <uint32_t Size>
inline __device__ glm::uvec3 GetLocalVoxelCoord(const glm::uvec3& centerVoxel, const glm::uvec3& voxel)
{
	static_assert(Size >= 3 && (Size - 1) % 2 == 0);
	constexpr int32_t distToOrigin = (Size - 1) / 2;
	glm::ivec3 diff = glm::ivec3(voxel) - glm::ivec3(centerVoxel);
	return glm::uvec3(diff + distToOrigin);
}

template <uint32_t Size>
inline __device__ bool ShouldProcessVoxel(uint32_t voxelID, const VCT::VoxelTraceData& voxelTraceData, const VCT::ConeTracingData& coneTracingData)
{
	glm::uvec3 voxelCoord = VCT::Utils::VoxelIDToCoord(voxelID, coneTracingData.voxelWorldInfo.dimensions);
	glm::uvec3 prevVoxelCoord = voxelTraceData.previousVoxel;

	glm::uvec3 localCoord = GetLocalVoxelCoord<3>(prevVoxelCoord, voxelCoord);

	bool inRange = localCoord.x < Size && localCoord.y < Size && localCoord.z < Size;
	return !inRange || voxelTraceData.voxelHistory[localCoord.x][localCoord.y][localCoord.z] != voxelID;
}

template <typename VoxelHandleFunc>
inline __device__ bool HandleKernel(VCT::PropagationData& parent,
									const glm::vec3& rayOrigin,
									const glm::vec3& currentVoxelCenter,
									const VCT::IntersectionData& intersectionData,
									const glm::uvec3& centerVoxel,
									const VCT::ConeTracingData& coneTracingData)
{
	constexpr uint32_t kernelSize = 3;
	constexpr uint32_t numVoxels = kernelSize * kernelSize * kernelSize;

	uint32_t kernelVoxelIDs[kernelSize][kernelSize][kernelSize];
	glm::u16vec3 start = parent.voxelTraceData.localVoxel;
	parent.voxelTraceData.localVoxel = glm::u16vec3(0);

	KernelFetchVoxelIDs(currentVoxelCenter, coneTracingData, kernelVoxelIDs);
	uint32_t nextHistoryBuffer[kernelSize][kernelSize][kernelSize];
	InitVoxelHistory(nextHistoryBuffer);

	//Unrolling this causes huge load time. PTX size increases to 5mb. 10% faster with it though.
	for (uint32_t i = 0; i < numVoxels; ++i)
	{
		glm::u16vec3 coord = VCT::Utils::VoxelIDToCoord(i, glm::uvec3(3));
		uint32_t voxelID = kernelVoxelIDs[coord.x][coord.y][coord.z];
		if (voxelID != VCT::Constants::InvalidPointIndex)
		{
			glm::uvec3 localVoxel = GetLocalVoxelCoord<kernelSize>(centerVoxel, VCT::Utils::VoxelIDToCoord(voxelID, coneTracingData.voxelWorldInfo.dimensions));
			nextHistoryBuffer[localVoxel.x][localVoxel.y][localVoxel.z] = voxelID;

			if (coord.x >= start.x && coord.y >= start.y && coord.z >= start.z)
			{
				start = glm::u16vec3(0);
				const VCT::VoxelInfo& voxelInfo = coneTracingData.voxelInfos[voxelID];

				constexpr float voxelBoundingSphereRadius = VCT::Constants::Sqrt3 / 2.0f;

				if (ShouldProcessVoxel<kernelSize>(voxelID, parent.voxelTraceData, coneTracingData) && intersectionData.Intersect(voxelInfo.voxelSpaceCenter, voxelBoundingSphereRadius, voxelBoundingSphereRadius))
				{
					if (!VoxelHandleFunc()(rayOrigin, voxelInfo, intersectionData, parent))
					{
						parent.voxelTraceData.localVoxel = coord;
						return false;
					}
				}
			}
		}
	}
	CopyVoxelHistory(parent.voxelTraceData.voxelHistory, nextHistoryBuffer);
	parent.voxelTraceData.previousVoxel = centerVoxel;
	return true;
}

template <typename VoxelHandleFunc>
inline __device__ void Propagate(VCT::PropagationData& parent,
								 const VCT::ConeTracingData& coneTracingData,
								 //VoxelHandleFunc voxelHandleFunc,
								 VCT::Status& status)
{
	VCT::VoxelTraceData& voxelTraceData = parent.voxelTraceData;
	const VCT::TraceData& traceData = parent.tpData.traceData;

	const glm::vec3& rayOrigin = traceData.interactions[traceData.numInteractions - 1].position;
	TextureTraverser traverser = TextureTraverser(voxelTraceData.voxel, voxelTraceData.rayDirection, coneTracingData.voxelTexture);

	do
	{
		if (traverser.GetMarchDistance() == 1)
		{
			if (!HandleKernel<VoxelHandleFunc>(parent, rayOrigin, traverser.GetTextureVoxel(), parent.voxelTraceData.intersectionData, glm::uvec3(traverser.GetCurrentVoxel()), coneTracingData))//, voxelHandleFunc))
			{
				voxelTraceData.voxel = traverser.GetTraverseVoxel();
				status = VCT::Status::ProcessingRequired;
				return;
			}
		}
		traverser.Step();
	} while (traverser.GetMarchDistance() != VCT::Constants::InvalidPointIndex);

	voxelTraceData.finished = true;
}