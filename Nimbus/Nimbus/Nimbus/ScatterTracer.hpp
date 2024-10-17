#pragma once
#include "Nimbus/PathStorage.hpp"
#include "PointCloudEnvironment.hpp"

namespace Nimbus
{
	class ScatterTracer
	{
	public:
		ScatterTracer();

		bool Prepare(const Environment& env,
					 const ScatterTracingParams& params,
					 const glm::vec3* txs,
					 uint32_t txCount,
					 const glm::vec3* rxs,
					 uint32_t rxCount);

		bool CreateCoverageMapInfo(const Environment& env, const glm::vec3& tx, float size, float height, CoverageMapInfo& result, std::vector<glm::vec3>& receivers);
		std::unique_ptr<PathStorage> Trace();

	private:
		void ComputeVisibility();
		void DetermineLOSPaths();
		void Transmit(uint32_t txID);
		void Propagate();
		void Refine();
		uint32_t RetrieveReceivedPaths();

	private:
		template <typename Type>
		struct PathData
		{
			PathData() = default;
			void Resize(uint32_t pathCount, uint32_t maxNumInteractions);

			DeviceBuffer pathInfoBuffer;
			DeviceBuffer interactionBuffer;
			DeviceBuffer normalBuffer;
			DeviceBuffer labelBuffer;
			DeviceBuffer materialBuffer;
		};

	private:
		const Environment* m_Environment;
		uint32_t m_IeCount;
		uint32_t m_TxCount;
		uint32_t m_RxCount;
		uint32_t m_MaxNumIa;
		bool m_Scattering;
		bool m_Diffraction;
		DeviceBuffer m_TransmitterBuffer;
		DeviceBuffer m_ReceiverBuffer;
		DeviceBuffer m_ScattererVisibleBuffer;
		DeviceBuffer m_RxVisibleBuffer;
		STRTData m_STRTData;
		DeviceBuffer m_STRTDataBuffer;
		DeviceBuffer m_PathProcessedBuffer;
		DeviceBuffer m_PathCountBuffer;
		DeviceBuffer m_ReceivedPathCountBuffer;
		uint32_t m_PropagationPathCount;
		PathData<PathInfo> m_ReceivedPathData;
		PathData<PathInfoST> m_PropagationPathData;
		std::vector<PathInfo> m_PathInfos;
		std::vector<glm::vec3> m_Interactions;
		std::vector<glm::vec3> m_Normals;
		std::vector<uint32_t> m_Labels;
		std::vector<uint32_t> m_Materials;
		std::unique_ptr<PathStorage> m_PathStorage;
	};

	template <typename Type>
	inline void ScatterTracer::PathData<Type>::Resize(uint32_t pathCount, uint32_t maxNumInteractions)
	{
		pathInfoBuffer = DeviceBuffer(sizeof(Type) * pathCount);
		if (maxNumInteractions > 0)
		{
			interactionBuffer = DeviceBuffer(sizeof(glm::vec3) * maxNumInteractions * pathCount);
			normalBuffer = DeviceBuffer(sizeof(glm::vec3) * maxNumInteractions * pathCount);
			labelBuffer = DeviceBuffer(sizeof(uint32_t) * maxNumInteractions * pathCount);
			materialBuffer = DeviceBuffer(sizeof(uint32_t) * maxNumInteractions * pathCount);
		}
	}
}