#pragma once

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <Traversal.hpp>
#include <texture_fetch_functions.h>


class TextureTraverser : public VCT::VoxelTraverser
{
public:
	__device__ TextureTraverser(const glm::vec3& voxelSpacePosition, const glm::vec3& rayDirection, cudaTextureObject_t voxelTexture);

	__device__ const uint2& GetTextureQueryResult() const;
	__device__ uint32_t GetVoxelID() const;
	__device__ uint32_t GetMarchDistance() const;
	__device__ uint32_t GetCurrentHits() const;
	__device__ void Step();

private:
	__device__ uint2 QueryTexture();

private:
	cudaTextureObject_t m_VoxelTexture;
	uint2 m_TextureQueryResult;
	uint32_t m_CurrentHits;
	uint32_t m_MaxHits;
};

inline __device__ TextureTraverser::TextureTraverser(const glm::vec3& voxelSpacePosition, const glm::vec3& rayDirection, cudaTextureObject_t voxelTexture)
	: VCT::VoxelTraverser(voxelSpacePosition, rayDirection)
	, m_VoxelTexture(voxelTexture)
	, m_TextureQueryResult(QueryTexture())
{

}

inline __device__ const uint2& TextureTraverser::GetTextureQueryResult() const
{
	return m_TextureQueryResult;
}

inline __device__ uint32_t TextureTraverser::GetVoxelID() const
{
	return m_TextureQueryResult.x;
}

inline __device__ uint32_t TextureTraverser::GetMarchDistance() const
{
	return m_TextureQueryResult.y;
}

inline __device__ uint32_t TextureTraverser::GetCurrentHits() const
{
	return m_CurrentHits;
}

inline __device__ void TextureTraverser::Step()
{
	VCT::VoxelTraverser::Step(GetMarchDistance() - 1);
	m_TextureQueryResult = QueryTexture();
	m_CurrentHits += GetVoxelID() != VCT::Constants::InvalidPointIndex;
}

inline __device__ uint2 TextureTraverser::QueryTexture()
{
	glm::vec3 texVoxel = GetTextureVoxel();
	return tex3D<uint2>(m_VoxelTexture, texVoxel.x, texVoxel.y, texVoxel.z);
}