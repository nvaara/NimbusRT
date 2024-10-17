#pragma once
#include <stdint.h>
#include <glm/gtc/constants.hpp>

namespace Nimbus::Constants
{
    constexpr uint32_t InvalidPointIndex = ~0u;
    constexpr uint32_t MaximumNumberOfInteractions = 8;
    constexpr float LightSpeedInVacuum = 299792458.0f;
    constexpr float InvLightSpeedInVacuum = 3.3356408746e-9f;
}