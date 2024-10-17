#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "Nimbus/Types.hpp"
#include "Nimbus/Utils.hpp"

__constant__ Nimbus::STData data;

inline __device__ void OptimizeAabbSize(OptixAabb& aabb, const glm::vec3& voxelCenter, float halfVoxelSize)
{
	aabb.minX = ((aabb.maxX - aabb.minX >= halfVoxelSize) ? voxelCenter.x - halfVoxelSize : aabb.minX) - data.aabbBias;
	aabb.maxX = ((aabb.maxX - aabb.minX >= halfVoxelSize) ? voxelCenter.x + halfVoxelSize : aabb.maxX) + data.aabbBias;
	
	aabb.minY = ((aabb.maxY - aabb.minY >= halfVoxelSize) ? voxelCenter.y - halfVoxelSize : aabb.minY) - data.aabbBias;
	aabb.maxY = ((aabb.maxY - aabb.minY >= halfVoxelSize) ? voxelCenter.y + halfVoxelSize : aabb.maxY) + data.aabbBias;
	
	aabb.minZ = ((aabb.maxZ - aabb.minZ >= halfVoxelSize) ? voxelCenter.z - halfVoxelSize : aabb.minZ) - data.aabbBias;
	aabb.maxZ = ((aabb.maxZ - aabb.minZ >= halfVoxelSize) ? voxelCenter.z + halfVoxelSize : aabb.maxZ) + data.aabbBias;
}

extern "C" __global__ void CreatePrimitives()
{
	uint32_t ieID = threadIdx.x + blockIdx.x * blockDim.x;
	if (ieID >= data.ieCount) return;
	
	glm::uvec2 nodeData = data.voxelPointNodeIndices[ieID];
	glm::vec3 world = Nimbus::Utils::VoxelIDToWorld(Nimbus::Utils::WorldToVoxelID(data.pointNodes[nodeData.x].position, data.voxelWorldInfo), data.voxelWorldInfo);
	uint32_t pointIndex = atomicAdd(data.pointCount, nodeData.y);
	uint32_t primitiveIndex = atomicAdd(data.primitiveCount, 1u);
	
	Nimbus::IEPrimitiveInfo& primitiveInfo = data.primitiveInfos[primitiveIndex];
	primitiveInfo.ID = primitiveIndex;
	primitiveInfo.pointIndexInfo.first = pointIndex;
	primitiveInfo.pointIndexInfo.count = nodeData.y;

	uint32_t nodeIndex = nodeData.x;
	glm::vec3 totalPos = glm::vec3(0.0f);
	OptixAabb& aabb = data.primitives[primitiveIndex];
	glm::vec3& rtPoint = data.rtPoints[primitiveIndex];

	aabb.minX = world.x + data.voxelWorldInfo.size;
	aabb.minY = world.y + data.voxelWorldInfo.size;
	aabb.minZ = world.z + data.voxelWorldInfo.size;

	aabb.maxX = world.x - data.voxelWorldInfo.size;
	aabb.maxY = world.y - data.voxelWorldInfo.size;
	aabb.maxZ = world.z - data.voxelWorldInfo.size;

	while (nodeIndex != Nimbus::Constants::InvalidPointIndex)
	{
		Nimbus::PointNode& node = data.pointNodes[nodeIndex];
		Nimbus::PrimitivePoint& point = data.points[pointIndex++];

		point.position = node.position;
		point.normal = node.normal;
		point.label = node.label;
		point.materialID = node.materialID;

		aabb.minX = glm::min(aabb.minX, node.position.x);
		aabb.minY = glm::min(aabb.minY, node.position.y);
		aabb.minZ = glm::min(aabb.minZ, node.position.z);

		aabb.maxX = glm::max(aabb.maxX, node.position.x);
		aabb.maxY = glm::max(aabb.maxY, node.position.y);
		aabb.maxZ = glm::max(aabb.maxZ, node.position.z);

		totalPos += point.position;
		nodeIndex = node.ieNext;
	}
	OptimizeAabbSize(aabb, world, data.voxelWorldInfo.halfSize);
	rtPoint = totalPos / static_cast<float>(nodeData.y);
}