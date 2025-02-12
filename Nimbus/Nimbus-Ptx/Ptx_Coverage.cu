#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <stdint.h>
#include "Nimbus/Types.hpp"
#include "Nimbus/Utils.hpp"

__constant__ Nimbus::CoverageData data;

extern "C" __global__ void CoveragePoints()
{
	uint32_t rtPointIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (rtPointIndex >= data.numPoints)
		return;

	glm::uvec3 cmVx = Nimbus::Utils::GetVoxelCoord(data.rtPoints[rtPointIndex], data.coverageWorldInfo);
	uint32_t rxID = cmVx.x + data.coverageWorldInfo.dimensions.x * cmVx.y;
	glm::uvec2 rxMask = Nimbus::Utils::GetBinBitMask32(rxID);
	uint32_t prev = atomicOr(&data.cellProcessed[rxMask.x], rxMask.y);

	if (!(prev & rxMask.y))
	{
		uint32_t index = atomicAdd(data.outNumRx, 1u);
		glm::vec3 rxPos = Nimbus::Utils::VoxelToWorld(glm::vec3(cmVx.x, cmVx.y, 0u), data.coverageWorldInfo);
		data.outReceivers[index] = glm::vec3(rxPos.x, rxPos.y, data.height);
		data.outRx2D[index] = glm::uvec2(cmVx.y, cmVx.x);
	}
}