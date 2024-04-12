#pragma once
#include <Types.hpp>
#include <unordered_map>
#include <string>

#define VEC3(name) union { glm::vec3 name; std::array<float, 3> name##Arr; }

namespace VCT
{
	struct SceneSettings
	{
		float frequency = 60e9f;
		float voxelSize = 0.5f;
		uint32_t voxelDivisionFactor = 2;
		uint32_t subvoxelDivisionFactor = 4;
		uint32_t receivedPathBufferSize = 25000;
		uint32_t propagationPathBufferSize = 100000;
		float propagationBufferSizeIncreaseFactor = 2.0f;

		float sampleRadiusCoarse = 0.015f;
		float sampleRadiusRefine = 0.003f;
		float varianceFactorCoarse = 2.0f;
		float varianceFactorRefine = 2.0f;
		float sdfThresholdCoarse = 0.0015f;
		float sdfThresholdRefine = 0.0005f;

		uint32_t numIterations = 2000;
		float delta = 1e-4f;
		float alpha = 0.4f;
		float beta = 0.4f;
		float angleThreshold = 1.0f;
		float distanceThreshold = 0.002f;

		uint32_t blockSize = 32;
		uint32_t numCoarsePathsPerUniqueRoute = 100;
	};

	struct Object3D
	{
		Object3D(const std::array<float, 3>& p)
			: position(p[0], p[1], p[2])
		{
		}

		glm::vec3 position;
	};

	using V3 = std::array<float, 3>;
	
	struct Edge
	{
		Edge(const V3& start,
			 const V3& end,
			 const V3& normal0,
			 const V3& normal1)
			: startArr(start)
			, endArr(end)
			, normal0Arr(normal0)
			, normal1Arr(normal1)
		{
		}

		VEC3(start);
		VEC3(end);
		VEC3(normal0);
		VEC3(normal1);
	};
	struct InputData
	{
		InputData()
			: numInteractions(0)
			, numDiffractions(0)
		{

		}

		SceneSettings sceneSettings;
		uint32_t numInteractions;
		uint32_t numDiffractions;
	};
}