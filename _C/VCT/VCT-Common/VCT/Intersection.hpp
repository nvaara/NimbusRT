#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

class Cone
{
public:
	struct IntersectionData
	{
		__device__ IntersectionData(bool valid, float radiusSq, float distanceSq);
		__device__ bool Intersection() const;

		bool valid;
		float radiusSq;
		float distanceSq;
	};

	__device__ Cone();
	__device__ Cone(const glm::vec3& origin, const glm::vec3& direction, float cosAngle);
	__device__ Cone(const glm::vec3& origin, const glm::vec3& direction, float cosAngle, float sinAngle);
	__device__ bool Intersect(const glm::vec3& point, float radiusOffset = 0.0f, float positionOffset = 0.0f) const;
	__device__ IntersectionData CalculateIntersectionData(const glm::vec3& point, float radiusOffset = 0.0f, float positionOffset = 0.0f) const;
	
	__device__ const glm::vec3& GetOrigin() const { return m_Origin; }
	__device__ const glm::vec3& GetDirection() const { return m_Direction; }
	__device__ float GetRadiusPerDirectionUnit() const { return m_RadiusPerDirectionUnit; }

private:
	glm::vec3 m_Origin;
	glm::vec3 m_Direction;
	float m_RadiusPerDirectionUnit;
};

inline __device__ Cone::IntersectionData::IntersectionData(bool valid, float radiusSq, float distanceSq)
	: valid(valid)
	, radiusSq(radiusSq)
	, distanceSq(distanceSq)
{

}

inline __device__ bool Cone::IntersectionData::Intersection() const
{ 
	return valid && distanceSq <= radiusSq;
}

inline __device__ Cone::Cone()
	: m_Origin({})
	, m_Direction({})
	, m_RadiusPerDirectionUnit(0.0f)
{
}

inline __device__ Cone::Cone(const glm::vec3& origin, const glm::vec3& direction, float cosAngle)
	: m_Origin(origin)
	, m_Direction(direction)
	, m_RadiusPerDirectionUnit(abs(sqrtf(1 - cosAngle * cosAngle) / glm::max(abs(cosAngle), 1e-9f)))
{
}

inline __device__ Cone::Cone(const glm::vec3& origin, const glm::vec3& direction, float cosAngle, float sinAngle)
	: m_Origin(origin)
	, m_Direction(direction)
	, m_RadiusPerDirectionUnit(abs(sinAngle / glm::max(abs(cosAngle), 1e-9f)))
{

}

inline __device__ bool Cone::Intersect(const glm::vec3& point, float radiusOffset, float positionOffset) const
{
	return CalculateIntersectionData(point, radiusOffset, positionOffset).Intersection();
}

inline __device__ Cone::IntersectionData Cone::CalculateIntersectionData(const glm::vec3& point, float radiusOffset, float positionOffset) const
{
	glm::vec3 vecToPoint = point - (m_Origin + m_Direction * positionOffset);
	float coneDist = glm::dot(vecToPoint, m_Direction);

	glm::vec3 distPointToCone = vecToPoint - coneDist * m_Direction;
	float distanceSq = glm::dot(distPointToCone, distPointToCone);
	float radius = coneDist * m_RadiusPerDirectionUnit + radiusOffset;
	float radiusSq = radius * radius;

	return IntersectionData(coneDist >= 0.0f, radiusSq, distanceSq);
}

class Plane
{
public:
	__device__ Plane();
	__device__ Plane(const glm::vec3& position, const glm::vec3& normal);
	__device__ float SignedDistance(const glm::vec3& p) const;

private:
	glm::vec3 m_Position;
	glm::vec3 m_Normal;
};

inline __device__ Plane::Plane()
	: m_Position(0.0f)
	, m_Normal(0.0f)
{
}

inline __device__ Plane::Plane(const glm::vec3& position, const glm::vec3& normal)
	: m_Position(position)
	, m_Normal(normal)
{
}

inline __device__ float Plane::SignedDistance(const glm::vec3& p) const
{
	return dot(p - m_Position, m_Normal);
}
