#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <stdint.h>
#include <Types.hpp>
#include <Utils.hpp>

__constant__ VCT::VoxelizationData data;

inline __device__ void OptimizeAabbSize(OptixAabb& aabb, const glm::vec3& voxelCenter, float halfVoxelSize)
{
	constexpr float bias = 0.0001f;

	aabb.minX = ((aabb.maxX - aabb.minX >= halfVoxelSize) ? voxelCenter.x - halfVoxelSize : aabb.minX) - bias;
	aabb.maxX = ((aabb.maxX - aabb.minX >= halfVoxelSize) ? voxelCenter.x + halfVoxelSize : aabb.maxX) + bias;

	aabb.minY = ((aabb.maxY - aabb.minY >= halfVoxelSize) ? voxelCenter.y - halfVoxelSize : aabb.minY) - bias;
	aabb.maxY = ((aabb.maxY - aabb.minY >= halfVoxelSize) ? voxelCenter.y + halfVoxelSize : aabb.maxY) + bias;

	aabb.minZ = ((aabb.maxZ - aabb.minZ >= halfVoxelSize) ? voxelCenter.z - halfVoxelSize : aabb.minZ) - bias;
	aabb.maxZ = ((aabb.maxZ - aabb.minZ >= halfVoxelSize) ? voxelCenter.z + halfVoxelSize : aabb.maxZ) + bias;
}

inline __device__ void WriteSurfaces(uint32_t surfaceVoxelID, uint32_t& surfaceIndex, uint32_t& primitiveIndex, uint32_t& pointIndex, uint32_t& edgeIndex, uint32_t& receiverIndex)
{
	uint2 nodeData = data.ieVoxelPointNodeIndices[surfaceVoxelID];
	uint32_t nodeIndex = nodeData.x;

	OptixAabb* aabb = nullptr;
	VCT::IntersectableEntity* ie = nullptr;
	VCT::IEPrimitiveInfo* primitiveInfo = nullptr;
	uint32_t label = VCT::Constants::InvalidPointIndex;
	float distSq = data.ieVoxelWorldInfo.size * 2.0f;
	distSq *= distSq;

	glm::vec3 world = VCT::Utils::VoxelIDToWorld(surfaceVoxelID, data.ieVoxelWorldInfo);
	if (nodeData.y > 0)
	{
		uint32_t pIdx = primitiveIndex++;
		uint32_t sIdx = surfaceIndex++;

		aabb = &data.iePrimitives[pIdx];
		aabb->minX = world.x + data.ieVoxelWorldInfo.size;
		aabb->minY = world.y + data.ieVoxelWorldInfo.size;
		aabb->minZ = world.z + data.ieVoxelWorldInfo.size;
		
		aabb->maxX = world.x - data.ieVoxelWorldInfo.size;
		aabb->maxY = world.y - data.ieVoxelWorldInfo.size;
		aabb->maxZ = world.z - data.ieVoxelWorldInfo.size;

		ie = &data.intersectableEntities[sIdx];
		ie->type = VCT::IEType::Surface;

		primitiveInfo = &data.iePrimitiveInfos[pIdx];
		primitiveInfo->pointIndexInfo.first = pointIndex;
		primitiveInfo->pointIndexInfo.count = nodeData.y;
		primitiveInfo->ID = sIdx;
	}

	while (nodeIndex != VCT::Constants::InvalidPointIndex)
	{
		VCT::PointNode& node = data.pointNodes[nodeIndex];
		switch (node.type)
		{
		case VCT::IEType::Receiver:
		{
			VCT::IntersectableEntity& rx = data.intersectableEntities[receiverIndex++];
			rx.type = node.type;
			rx.rtPoint = node.position;
			rx.voxelSpaceRtPoint = VCT::Utils::WorldToVoxel(rx.rtPoint, data.voxelWorldInfo);
			rx.receiverID = node.receiverID;
			break;
		}
		case VCT::IEType::Edge:
		{
			VCT::IntersectableEntity& edge = data.intersectableEntities[edgeIndex++];
			edge.type = node.type;
			edge.rtPoint = node.position;
			edge.voxelSpaceRtPoint = VCT::Utils::WorldToVoxel(edge.rtPoint, data.voxelWorldInfo);
			edge.edgeSegmentID = node.edgeSegmentID;
			break;
		}
		case VCT::IEType::Surface:
		{
			aabb->minX = min(aabb->minX, node.position.x);
			aabb->minY = min(aabb->minY, node.position.y);
			aabb->minZ = min(aabb->minZ, node.position.z);
				
			aabb->maxX = max(aabb->maxX, node.position.x);
			aabb->maxY = max(aabb->maxY, node.position.y);
			aabb->maxZ = max(aabb->maxZ, node.position.z);
			
			VCT::PrimitivePoint& point = data.iePrimitivePoints[pointIndex++];
			point.position = node.position;
			point.normal = node.normal;
			point.label = node.label;
			point.materialID = node.materialID;

			float cDistSq = glm::dot(point.position - world, point.position - world);
			label = cDistSq < distSq ? node.label : label;
			distSq = cDistSq < distSq ? cDistSq : distSq;
			break;
		}
		}
		nodeIndex = node.ieNext;
	}
	
	if (nodeData.y > 0)
	{
		glm::vec3 totalPos = glm::vec3(0.0f);
		for (uint32_t pIdx = 0; pIdx < primitiveInfo->pointIndexInfo.count; ++pIdx)
		{
			VCT::PrimitivePoint& pp = data.iePrimitivePoints[primitiveInfo->pointIndexInfo.first + pIdx];
			totalPos += pp.position;
		}		
		
		OptimizeAabbSize(*aabb, world, data.ieVoxelWorldInfo.halfSize);
		ie->rtPoint = totalPos / static_cast<float>(primitiveInfo->pointIndexInfo.count);
		ie->voxelSpaceRtPoint = VCT::Utils::WorldToVoxel(ie->rtPoint, data.voxelWorldInfo);
		ie->surfaceLabel = label;
	}
}

extern "C" __global__ void VoxelizePointCloud()
{
	uint32_t voxelID = threadIdx.x + blockIdx.x * blockDim.x;
	if (voxelID >= data.voxelWorldInfo.count) return;

	uint32_t voxelFactor = data.ieVoxelFactor;
	
	glm::vec3 voxelOriginWs = VCT::Utils::VoxelIDToWorld(voxelID, data.voxelWorldInfo) - data.voxelWorldInfo.halfSize;
	glm::uvec3 ieVoxelOrigin = VCT::Utils::GetVoxelCoord(voxelOriginWs + data.ieVoxelWorldInfo.halfSize, data.ieVoxelWorldInfo);

	const VCT::VoxelPointData& vpData = data.voxelPointData[voxelID];
	uint32_t ieTotalCount = vpData.numPrimitives + vpData.numReceivers + vpData.numEdges;
	VCT::VoxelInfo& voxelInfo = data.voxelInfos[voxelID];
	voxelInfo.voxelSpaceCenter = glm::vec3(0.5f) + glm::vec3(VCT::Utils::VoxelIDToCoord(voxelID, data.voxelWorldInfo.dimensions));
	voxelInfo.ieIndexInfo.first = atomicAdd(data.ieCount, ieTotalCount);
	voxelInfo.ieIndexInfo.count = ieTotalCount;
	
	uint32_t firstSurfaceIndex = voxelInfo.ieIndexInfo.first;
	uint32_t firstPrimitiveIndex = atomicAdd(data.iePrimitiveCount, vpData.numPrimitives);
	uint32_t firstPointIndex = atomicAdd(data.iePointCount, vpData.numSurfacePoints);
	uint32_t firstEdgeIndex = voxelInfo.ieIndexInfo.first + vpData.numPrimitives;
	uint32_t firstReceiverIndex = firstEdgeIndex + vpData.numEdges;

	for (uint32_t x = 0; x < voxelFactor; ++x)
	{
		for (uint32_t y = 0; y < voxelFactor; ++y)
		{
			for (uint32_t z = 0; z < voxelFactor; ++z)
			{
				glm::uvec3 coord = ieVoxelOrigin + glm::uvec3(x, y, z);
				uint32_t surfaceVoxelID = VCT::Utils::VoxelCoordToID(coord, data.ieVoxelWorldInfo.dimensions);
				WriteSurfaces(surfaceVoxelID, firstSurfaceIndex, firstPrimitiveIndex, firstPointIndex, firstEdgeIndex, firstReceiverIndex);
			}
		}
	}
}

inline __device__ bool WriteRefinePrimitiveData(const glm::vec3& voxelCenter, const VCT::VoxelWorldInfo& vwInfo, const VCT::IEPrimitiveInfo& ieInfo, uint32_t& primitiveIndex, uint32_t& primitivePointIndex)
{
	uint32_t pointsInPrimitive = 0;
	uint32_t first = primitivePointIndex;
	glm::vec3 minPos = voxelCenter + vwInfo.halfSize;
	glm::vec3 maxPos = voxelCenter - vwInfo.halfSize;
	uint32_t voxelID = VCT::Utils::WorldToVoxelID(voxelCenter, vwInfo);

	//Experimental
	glm::vec3 normalTotal = glm::vec3(0.0f);
	glm::vec3 posTotal = glm::vec3(0.0f);
	//

	for (uint32_t localIndex = 0; localIndex < ieInfo.pointIndexInfo.count; ++localIndex)
	{
		const VCT::PrimitivePoint& point = data.iePrimitivePoints[ieInfo.pointIndexInfo.first + localIndex];
		if (voxelID == VCT::Utils::WorldToVoxelID(point.position, vwInfo))
		{
			data.subIePrimitivePoints[primitivePointIndex++] = point;
			++pointsInPrimitive;
			minPos = glm::min(minPos, point.position);
			maxPos = glm::max(maxPos, point.position);
			//Experimental
			normalTotal += point.normal;
			posTotal += point.position;
			//
		}
	}

	if (pointsInPrimitive > 0)
	{
		constexpr float bias = 0.0001f;
		OptixAabb aabb{};
		aabb.minX = minPos.x;
		aabb.minY = minPos.y;
		aabb.minZ = minPos.z;

		aabb.maxX = maxPos.x;
		aabb.maxY = maxPos.y;
		aabb.maxZ = maxPos.z;

		aabb.minX = ((aabb.maxX - aabb.minX >= vwInfo.halfSize) ? voxelCenter.x - vwInfo.halfSize : aabb.minX) - bias;
		aabb.maxX = ((aabb.maxX - aabb.minX >= vwInfo.halfSize) ? voxelCenter.x + vwInfo.halfSize : aabb.maxX) + bias;
		
		aabb.minY = ((aabb.maxY - aabb.minY >= vwInfo.halfSize) ? voxelCenter.y - vwInfo.halfSize : aabb.minY) - bias;
		aabb.maxY = ((aabb.maxY - aabb.minY >= vwInfo.halfSize) ? voxelCenter.y + vwInfo.halfSize : aabb.maxY) + bias;
		
		aabb.minZ = ((aabb.maxZ - aabb.minZ >= vwInfo.halfSize) ? voxelCenter.z - vwInfo.halfSize : aabb.minZ) - bias;
		aabb.maxZ = ((aabb.maxZ - aabb.minZ >= vwInfo.halfSize) ? voxelCenter.z + vwInfo.halfSize : aabb.maxZ) + bias;

		uint32_t pIdx = primitiveIndex++;
		data.subIePrimitives[pIdx] = aabb;
		VCT::IEPrimitiveInfo& info = data.subIePrimitiveInfos[pIdx];
		info.pointIndexInfo.first = first;
		info.pointIndexInfo.count = pointsInPrimitive;
		info.ID = ieInfo.ID;
	}
	return pointsInPrimitive > 0;
}

extern "C" __global__ void WriteRefineAabb()
{
	uint32_t iePrimitiveID = threadIdx.x + blockIdx.x * blockDim.x;
	if (iePrimitiveID >= *data.iePrimitiveCount) return;
	uint32_t rvFactor = data.subIe;
	const VCT::IEPrimitiveInfo& ieInfo = data.iePrimitiveInfos[iePrimitiveID];

	uint32_t ieVoxelID = VCT::Utils::WorldToVoxelID(data.intersectableEntities[ieInfo.ID].rtPoint, data.ieVoxelWorldInfo);
	glm::vec3 voxelOriginWs = VCT::Utils::VoxelIDToWorld(ieVoxelID, data.ieVoxelWorldInfo) - data.ieVoxelWorldInfo.halfSize;
	glm::vec3 refineVoxelWs = voxelOriginWs + data.subIeVoxelWorldInfo.halfSize;

	uint32_t primitiveIndex = atomicAdd(data.subIePrimitiveCount, data.perIeSubIePrimitiveCount[ieVoxelID]);
	uint32_t primitivePointIndex = atomicAdd(data.subIePrimitivePointCount, ieInfo.pointIndexInfo.count);
	
	for (uint32_t x = 0; x < rvFactor; ++x)
	{
		for (uint32_t y = 0; y < rvFactor; ++y)
		{
			for (uint32_t z = 0; z < rvFactor; ++z)
			{
				uint32_t pIdx = primitiveIndex;
				glm::vec3 wpos = glm::vec3(refineVoxelWs.x + data.subIeVoxelWorldInfo.size * x,
										   refineVoxelWs.y + data.subIeVoxelWorldInfo.size * y,
										   refineVoxelWs.z + data.subIeVoxelWorldInfo.size * z);
				if (WriteRefinePrimitiveData(wpos, data.subIeVoxelWorldInfo, ieInfo, primitiveIndex, primitivePointIndex))
				{
					uint32_t refineVoxelID = VCT::Utils::WorldToVoxelID(wpos, data.subIeVoxelWorldInfo);
					data.subIePrimitiveVoxelMap[refineVoxelID] = pIdx;
				}
			}
		}
	}
}

extern "C" __global__ void WriteRefinePrimitiveNeighbors()
{
	uint32_t refinePrimitiveID = threadIdx.x + blockIdx.x * blockDim.x;
	if (refinePrimitiveID >= *data.subIePrimitiveCount)
		return;

	uint32_t refineVoxelID = VCT::Utils::WorldToVoxelID(data.subIePrimitivePoints[data.subIePrimitiveInfos[refinePrimitiveID].pointIndexInfo.first].position, data.subIeVoxelWorldInfo);
	VCT::PrimitiveNeighbors& primitiveNeighbors = data.subIePrimitiveNeighbors[refinePrimitiveID];
	primitiveNeighbors.count = 0;
	glm::uvec3 dimensions = data.subIeVoxelWorldInfo.dimensions;
	glm::ivec3 refineVoxelCoord = static_cast<glm::ivec3>(VCT::Utils::VoxelIDToCoord(refineVoxelID, dimensions));

	#pragma unroll
	for (int32_t x = -1; x <= 1; ++x)
	{
		#pragma unroll
		for (int32_t y = -1; y <= 1; ++y)
		{
			#pragma unroll
			for (int32_t z = -1; z <= 1; ++z)
			{
				glm::ivec3 curCoord = refineVoxelCoord + glm::ivec3(x, y, z);
				bool isCenter = x == 0 && y == 0 && z == 0;
				bool validLowerBound = curCoord.x >= 0 && curCoord.y >= 0 && curCoord.z >= 0;
				bool validUpperBound = curCoord.x < dimensions.x && curCoord.y < dimensions.y && curCoord.z < dimensions.z;

				if (!isCenter && validLowerBound && validUpperBound)
				{
					uint32_t neighborVoxelID = VCT::Utils::VoxelCoordToID(static_cast<glm::uvec3>(curCoord), dimensions);
					uint32_t neighborPrimitiveID = data.subIePrimitiveVoxelMap[neighborVoxelID];
					if (neighborPrimitiveID != VCT::Constants::InvalidPointIndex)
						primitiveNeighbors.neighbors[primitiveNeighbors.count++] = neighborPrimitiveID;
				}
			}
		}
	}
}

inline __device__ bool SampleX(const int3& coord, int32_t range)
{
	const glm::vec3& dimensions = data.voxelWorldInfo.dimensions;
	for (int32_t currentX = coord.x - range; currentX <= coord.x + range; currentX += range * 2)
	{
		if (currentX >= 0 && currentX < dimensions.x)
		{
			for (int32_t currentY = coord.y - range; currentY <= coord.y + range; ++currentY)
			{
				if (currentY >= 0 && currentY < dimensions.y)
				{
					for (int32_t currentZ = coord.z - range; currentZ <= coord.z + range; ++currentZ)
					{
						if (currentZ >= 0 && currentZ < dimensions.z)
						{
							uint32_t voxelIndex = VCT::Utils::VoxelCoordToID(glm::uvec3(currentX, currentY, currentZ), dimensions);
							if (data.voxelTextureData[voxelIndex].x != VCT::Constants::InvalidPointIndex)
								return true;
						}
					}
				}
			}
		}
	}
	return false;
}

inline __device__ bool SampleY(const int3& coord, int32_t range)
{
	const glm::vec3& dimensions = data.voxelWorldInfo.dimensions;
	for (int32_t currentY = coord.y - range; currentY <= coord.y + range; currentY += range * 2)
	{
		if (currentY >= 0 && currentY < dimensions.y)
		{
			for (int32_t currentX = coord.x - range; currentX <= coord.x + range; ++currentX)
			{
				if (currentX >= 0 && currentX < dimensions.x)
				{
					for (int32_t currentZ = coord.z - range; currentZ <= coord.z + range; ++currentZ)
					{
						if (currentZ >= 0 && currentZ < dimensions.z)
						{
							uint32_t voxelIndex = VCT::Utils::VoxelCoordToID(glm::uvec3(currentX, currentY, currentZ), dimensions);
							if (data.voxelTextureData[voxelIndex].x != VCT::Constants::InvalidPointIndex)
								return true;
						}
					}
				}
			}
		}
	}
	return false;
}

inline __device__ bool SampleZ(const int3& coord, int32_t range)
{
	const glm::vec3& dimensions = data.voxelWorldInfo.dimensions;
	for (int32_t currentZ = coord.z - range; currentZ <= coord.z + range; currentZ += range * 2)
	{
		if (currentZ >= 0 && currentZ < dimensions.z)
		{
			for (int32_t currentX = coord.x - range; currentX <= coord.x + range; ++currentX)
			{
				if (currentX >= 0 && currentX < dimensions.x)
				{
					for (int32_t currentY = coord.y - range; currentY <= coord.y + range; ++currentY)
					{
						if (currentY >= 0 && currentY < dimensions.y)
						{
							uint32_t voxelIndex = VCT::Utils::VoxelCoordToID(glm::uvec3(currentX, currentY, currentZ), dimensions);
							if (data.voxelTextureData[voxelIndex].x != VCT::Constants::InvalidPointIndex)
								return true;
						}
					}
				}
			}
		}
	}
	return false;
}

extern "C" __global__ void FillTextureData()
{
	uint32_t voxelIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (voxelIndex < data.voxelWorldInfo.count)
	{
		const glm::vec3& dimensions = data.voxelWorldInfo.dimensions;
		uint2& texData = data.voxelTextureData[voxelIndex];
		uint32_t firstPointNode = texData.x;
		int32_t range = 1;
		if (firstPointNode != VCT::Constants::InvalidPointIndex)
		{
			texData.y = range;
			return;
		}
		glm::uvec3 ucoord = VCT::Utils::VoxelIDToCoord(voxelIndex, dimensions);
		int3 coord = make_int3(ucoord.x, ucoord.y, ucoord.z);

		while (true)
		{
			if (SampleX(coord, range))
				break;

			if (SampleY(coord, range))
				break;
			
			if (SampleZ(coord, range))
				break;

			int32_t xOutOfRange = (coord.x + range >= dimensions.x) & (coord.x - range < 0);
			int32_t yOutOfRange = (coord.y + range >= dimensions.y) & (coord.y - range < 0);
			int32_t zOutOfRange = (coord.z + range >= dimensions.z) & (coord.z - range < 0);
			if (xOutOfRange && yOutOfRange && zOutOfRange)
			{
				range = max(max(dimensions.x, dimensions.y), dimensions.z);
				break;
			}
			++range;
		}
		texData.y = range;
	}
}