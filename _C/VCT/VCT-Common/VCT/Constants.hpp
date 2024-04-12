#pragma once
#include <stdint.h>
#include <glm/gtc/constants.hpp>

namespace VCT::Constants
{
    constexpr uint32_t InvalidPointIndex = ~0u;
    constexpr uint32_t MaximumNumberOfInteractions = 8;
    constexpr uint32_t UnitCircleDiscretizationCount = 100;

    constexpr float SeparationPlaneBias = 1e-2f;
    constexpr float LightSpeedInVacuum = 299792458.0f;
    constexpr float InvLightSpeedInVacuum = 3.3356408746e-9f;
    constexpr float Pi = 3.1415926535897932384626433832795f;
    constexpr float RootPi = 1.7724538509055160272981674833411f;
    constexpr float ImpedanceOfFreeSpace = 376.73031346f;

    constexpr float Sqrt2 = 1.4142135623730950488016887242097f;
    constexpr float Sqrt3 = 1.7320508075688772935274463415059f;
}