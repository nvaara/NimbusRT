#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "Nimbus/Types.hpp"
#include "Nimbus/Utils.hpp"

inline __device__ bool TestAxis(const glm::vec3& axis, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& extent)
{
	float proj0 = glm::dot(v0, axis);
	float proj1 = glm::dot(v1, axis);
	float proj2 = glm::dot(v2, axis);

	float r = extent.x * glm::abs(glm::dot(glm::vec3(1.0f, 0.0f, 0.0f), axis)) +
		extent.y * glm::abs(glm::dot(glm::vec3(0.0f, 1.0f, 0.0f), axis)) +
		extent.z * glm::abs(glm::dot(glm::vec3(0.0f, 0.0f, 1.0f), axis));

	return glm::max(-glm::max(proj0, glm::max(proj1, proj2)), glm::min(proj0, glm::min(proj1, proj2))) <= r;
}

inline __device__ bool TriangleIntersectsVoxel(const glm::vec3& voxelCenterWs, const glm::vec3& extent, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2)
{
	glm::vec3 v0 = p0 - voxelCenterWs;
	glm::vec3 v1 = p1 - voxelCenterWs;
	glm::vec3 v2 = p2 - voxelCenterWs;

	glm::vec3 f0 = v1 - v0;
	glm::vec3 f1 = v2 - v1;
	glm::vec3 f2 = v0 - v2;

	glm::vec3 u0 = glm::vec3(1.0f, 0.0f, 0.0f);
	glm::vec3 u1 = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::vec3 u2 = glm::vec3(0.0f, 0.0f, 1.0f);

	glm::vec3 u0f0 = glm::cross(u0, f0);
	glm::vec3 u0f1 = glm::cross(u0, f1);
	glm::vec3 u0f2 = glm::cross(u0, f2);

	glm::vec3 u1f0 = glm::cross(u1, f0);
	glm::vec3 u1f1 = glm::cross(u1, f1);
	glm::vec3 u1f2 = glm::cross(u2, f2);

	glm::vec3 u2f0 = glm::cross(u2, f0);
	glm::vec3 u2f1 = glm::cross(u2, f1);
	glm::vec3 u2f2 = glm::cross(u2, f2);

	return TestAxis(u0f0, v0, v1, v2, extent)
		&& TestAxis(u0f1, v0, v1, v2, extent)
		&& TestAxis(u0f2, v0, v1, v2, extent)
		&& TestAxis(u1f0, v0, v1, v2, extent)
		&& TestAxis(u1f1, v0, v1, v2, extent)
		&& TestAxis(u1f2, v0, v1, v2, extent)
		&& TestAxis(u2f0, v0, v1, v2, extent)
		&& TestAxis(u2f1, v0, v1, v2, extent)
		&& TestAxis(u2f2, v0, v1, v2, extent)
		&& TestAxis(u0, v0, v1, v2, extent)
		&& TestAxis(u1, v0, v1, v2, extent)
		&& TestAxis(u2, v0, v1, v2, extent)
		&& TestAxis(glm::cross(f0, f1), v0, v1, v2, extent);
}

inline __device__ void CreatePrimitives(const Nimbus::TriangleData* data, uint32_t faceIndex)
{
	glm::uvec3 vertexIndices = data->indices[faceIndex];
	glm::vec3 p0 = data->vertices[vertexIndices.x];
	glm::vec3 p1 = data->vertices[vertexIndices.y];
	glm::vec3 p2 = data->vertices[vertexIndices.z];

	const glm::vec3& normal = data->faces[faceIndex].normal;
	glm::vec3 minAabb = glm::min(glm::min(p0, p1), p2);
	glm::vec3 maxAabb = glm::max(glm::max(p0, p1), p2);

	glm::uvec3 minVoxel = Nimbus::Utils::VoxelIDToCoord(Nimbus::Utils::WorldToVoxelID(minAabb, data->vwInfo), data->vwInfo.dimensions);
	glm::uvec3 maxVoxel = Nimbus::Utils::VoxelIDToCoord(Nimbus::Utils::WorldToVoxelID(maxAabb, data->vwInfo), data->vwInfo.dimensions);
	constexpr float bias = 1e-4f;
	glm::vec3 voxelExtent = glm::vec3(data->vwInfo.halfSize + bias);

	for (uint32_t x = minVoxel.x; x <= maxVoxel.x; ++x)
	{
		for (uint32_t y = minVoxel.y; y <= maxVoxel.y; ++y)
		{
			for (uint32_t z = minVoxel.z; z <= maxVoxel.z; ++z)
			{
				glm::uvec3 voxelCoord = glm::uvec3(x, y, z);
				glm::vec3 voxelCenterWs = Nimbus::Utils::VoxelToWorld(voxelCoord, data->vwInfo);
				if (TriangleIntersectsVoxel(voxelCenterWs, voxelExtent, p0, p1, p2))
				{
					uint32_t voxelID = Nimbus::Utils::VoxelCoordToID(voxelCoord, data->vwInfo.dimensions);
					glm::vec3 projectionAxisDistance = glm::dot(voxelCenterWs - p0, normal) * normal;
					glm::vec3 absAxisDist = glm::abs(projectionAxisDistance);
					glm::vec3 projectedPoint = voxelCenterWs - projectionAxisDistance;
					bool projectedPointInVoxel = (absAxisDist.x <= voxelExtent.x
											   && absAxisDist.y <= voxelExtent.y
											   && absAxisDist.z <= voxelExtent.z);

					if (projectedPointInVoxel)
					{
						constexpr uint32_t temporaryValue = 0u;
						uint32_t oldValue = atomicCAS(&data->voxelToRtPointIndexMap[voxelID], Nimbus::Constants::InvalidPointIndex, temporaryValue);
						if (oldValue == Nimbus::Constants::InvalidPointIndex)
						{
							uint32_t rtPointIndex = atomicAdd(data->rtPointCounter, 1u);
							data->voxelToRtPointIndexMap[voxelID] = rtPointIndex;
							data->rtPoints[rtPointIndex] = projectedPoint;
						}
					}
				}
			}
		}
	}
}

extern "C" __global__ void CreateTrianglePrimitives(const Nimbus::TriangleData* data)
{
	uint32_t faceIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (faceIndex >= data->numFaces) return;
	CreatePrimitives(data, faceIndex);
}

extern "C" __global__ void CreateTriangleCoverageMap()
{

}