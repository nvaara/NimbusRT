#include "Hash.hpp"

namespace Nimbus
{
	namespace
	{
		bool operator==(const PathInfo& lhs, const PathInfo& rhs)
		{
			return lhs.txID == rhs.txID
				&& lhs.rxID == rhs.rxID
				&& lhs.pathType == rhs.pathType
				&& lhs.numInteractions == rhs.numInteractions;
		}
	}

	PathHash::PathHash(const PathInfo& pathInfo, const uint32_t* labels)
		: m_PathInfo(pathInfo)
		, m_Labels({})
		, m_HashKey(0ull)
	{
		uint64_t hashableTxID = (static_cast<uint64_t>(pathInfo.pathType) << 32u) | pathInfo.txID;
		uint64_t hashableRxID = (static_cast<uint64_t>(pathInfo.numInteractions + 1u) << 32u) | pathInfo.rxID;
		m_HashKey = Utils::CalculateHash(hashableTxID, hashableRxID);

		for (uint32_t i = 0; i < pathInfo.numInteractions; ++i)
		{
			m_Labels[i] = labels[i];
			Utils::CombineHash(m_HashKey, (static_cast<uint64_t>(i + 1u) << 32u) | labels[i]);
		}
	}

	bool PathHash::AreLabelsEqual(const PathHash& other) const
	{
		for (uint32_t i = 0; i < m_PathInfo.numInteractions; ++i)
		{
			if (m_Labels[i] != other.m_Labels[i])
				return false;
		}
		return true;
	}

	bool PathHash::operator==(const PathHash& other) const
	{
		return (m_HashKey == other.m_HashKey
			 && m_PathInfo == other.m_PathInfo
			 && AreLabelsEqual(other));
	}
}