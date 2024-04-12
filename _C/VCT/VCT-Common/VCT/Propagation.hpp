#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "Constants.hpp"
#ifndef __CUDACC__
#include <array>
#endif

namespace VCT
{
    struct Angle
    {
        __device__ Angle() : azimuth(0.0f), elevation(0.0f) {}
        __device__ Angle(float azimuth, float elevation) : azimuth(azimuth), elevation(elevation) {}
        __device__ Angle(const glm::vec3& direction);

        __device__ glm::vec3 MapToDirectionVector() const;

        float azimuth;
        float elevation;
    };

    inline __device__ Angle::Angle(const glm::vec3& direction)
        : azimuth(std::atan2(direction.y, direction.x))
        , elevation(std::acos(direction.z))
    {
    }

    inline __device__ glm::vec3 Angle::MapToDirectionVector() const
    {
        float cosEl = std::cos(elevation);
        return glm::vec3(std::cos(azimuth) * cosEl, std::sin(azimuth) * cosEl, std::sin(elevation));
    }

    struct Transmitter
    {
        __device__ Transmitter() = default;
        __device__ Transmitter(const glm::vec3& position);

        glm::vec3 position;
    };

    inline __device__ Transmitter::Transmitter(const glm::vec3& position)
        : position(position)
    {
    }

    struct Receiver
    {
        __device__ Receiver() = default;
        __device__ Receiver(const glm::vec3& position) : position(position) {}
        glm::vec3 position;
    };

    enum class InteractionType : uint32_t
    {
        Diffraction = 1,
        Reflection = 2,
    };

    struct Interaction
    {
        uint32_t ieID;
        uint32_t hitID;
        uint32_t label;
        InteractionType type;
        union
        {
            glm::vec3 position;
#ifndef __CUDACC__
            std::array<float, 3> posArr;
#endif
        };

        union
        {
            glm::vec3 normal;
#ifndef __CUDACC__
            std::array<float, 3> normArr;
#endif
        };
        float curvature;
        uint32_t materialID;
    };

    struct TraceData
    {
        uint32_t transmitterID;
        uint32_t receiverID;
        uint32_t numInteractions;
        float timeDelay;
#ifdef __CUDACC__
        Interaction interactions[Constants::MaximumNumberOfInteractions];
#else
        std::array<Interaction, Constants::MaximumNumberOfInteractions> interactions;
#endif
    };
}