#include "Environment.hpp"
#include "Nimbus/Utils.hpp"
#include "CudaUtils.hpp"
#include <array>

namespace Nimbus
{
    bool Environment::InitRisGasData(const RisData& risData)
    {
        m_RisData = {};
        size_t numRis = risData.objectIds.size();
        if (numRis)
        {
            std::vector<glm::vec3> vertices;
            std::vector<uint32_t> indices;
            vertices.reserve(4 * numRis);
            indices.reserve(6 * numRis);
            uint32_t indexOffset = 0u;
            for (uint32_t risIndex = 0; risIndex < risData.centers.size(); ++risIndex)
            {
                glm::vec3 u{}, v{};
                glm::vec3 center = risData.centers[risIndex];
                glm::vec2 halfSize = risData.size[risIndex] * 0.5f;
                glm::vec3 normal = risData.normals[risIndex];
                Nimbus::Utils::GetOrientationVectors(risData.normals[risIndex], u, v);

                vertices.push_back(center - halfSize.x * u - halfSize.y * v);
                vertices.push_back(center + halfSize.x * u - halfSize.y * v);
                vertices.push_back(center + halfSize.x * u + halfSize.y * v);
                vertices.push_back(center - halfSize.x * u + halfSize.y * v);

                indices.push_back(0 + indexOffset);
                indices.push_back(1 + indexOffset);
                indices.push_back(2 + indexOffset);
                indices.push_back(2 + indexOffset);
                indices.push_back(3 + indexOffset);
                indices.push_back(0 + indexOffset);

                indexOffset += 4;
            }

            m_RisData.vertexBuffer = DeviceBuffer::Create(vertices);
            m_RisData.indexBuffer = DeviceBuffer::Create(indices);
            m_RisData.objectIds = DeviceBuffer::Create(risData.objectIds);
            m_RisData.cellObjectIds = DeviceBuffer::Create(risData.cellObjectIds);
            m_RisData.cellWorldPositions = DeviceBuffer::Create(risData.cellWorldPositions);
            m_RisData.normals = DeviceBuffer::Create(risData.normals);
            m_RisData.cellCount = static_cast<uint32_t>(risData.cellObjectIds.size());
            m_RisData.gas = AccelerationStructure::CreateFromTriangles(m_RisData.vertexBuffer, static_cast<uint32_t>(vertices.size()), m_RisData.indexBuffer, static_cast<uint32_t>(indices.size() / 3u));
        }
        return m_RisData.gas != OptixTraversableHandle(0ull);
    }

    OptixTraversableHandle Environment::GetAccelerationStructure()
    {
        if (m_RisData.gas)
        {
            std::array<OptixInstance, 2> instances = {};
            glm::mat3x4 identityMatrix = glm::mat3x4(1.0f);

            instances[0].instanceId = 0;
            instances[0].sbtOffset = 0;
            instances[0].visibilityMask = 255u;
            instances[0].flags = OPTIX_INSTANCE_FLAG_NONE;
            std::memcpy(instances[0].transform, &identityMatrix, sizeof(glm::mat3x4));
            instances[0].traversableHandle = m_AccelerationStructure.GetRawHandle();

            instances[1].instanceId = 1;
            instances[1].sbtOffset = 1;
            instances[1].visibilityMask = 255u;
            instances[1].flags = OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
            std::memcpy(instances[1].transform, &identityMatrix, sizeof(glm::mat3x4));
            instances[1].traversableHandle = m_RisData.gas.GetRawHandle();

            m_InstanceBuffer = DeviceBuffer::Create(instances.data(), 2u);
            m_InstanceAs = AccelerationStructure::CreateFromInstances(m_InstanceBuffer, 2u);

            return m_InstanceAs.GetRawHandle();
        }
        return m_AccelerationStructure.GetRawHandle();
    }

    bool Environment::ProcessEdges(const EdgeData* edges, size_t numEdges)
    {
        m_Edges.reserve(numEdges);
        for (size_t i = 0; i < numEdges; ++i)
        {
            const EdgeData& edgeData = edges[i];
            DiffractionEdge edge{};
            glm::vec3 up{}, right{};
            edge.forward = glm::normalize(edgeData.end - edgeData.start);
            Utils::GetOrientationVectors(edge.forward, right, up);
            edge.start = edgeData.start;
            edge.end = edgeData.end;
            edge.halfLength = glm::length(edge.start - edge.end) * 0.5f;
            edge.midPoint = edge.start + edge.forward * edge.halfLength;
            edge.normal0 = edgeData.normal1;
            edge.normal1 = edgeData.normal2;
            edge.inverseMatrix = glm::transpose(glm::mat3(right, edge.forward, up));

            glm::vec3 lerpNormal = glm::normalize(glm::mix(edge.normal0, edge.normal1, 0.5f));
            glm::vec3 n0 = glm::normalize(glm::cross(edge.forward, edge.normal0));
            glm::vec3 n1 = glm::normalize(glm::cross(edge.forward, edge.normal1));
            n0 = glm::dot(lerpNormal, n0) < 0.0f ? n0 : -n0;
            n1 = glm::dot(lerpNormal, n1) < 0.0f ? n1 : -n1;
            glm::vec3 localSurfaceDir2D0 = edge.inverseMatrix * n0;
            glm::vec3 localSurfaceDir2D1 = edge.inverseMatrix * n1;

            edge.localSurfaceDir2D0 = glm::normalize(glm::vec2(localSurfaceDir2D0.x, localSurfaceDir2D0.z));
            edge.localSurfaceDir2D1 = glm::normalize(glm::vec2(localSurfaceDir2D1.x, localSurfaceDir2D1.z));

            m_Edges.push_back(edge);
        }
        m_EdgeBuffer = DeviceBuffer::Create(m_Edges);
        return m_EdgeBuffer.GetRawHandle() != CUdeviceptr(0); m_Edges.reserve(numEdges);
    }
}
