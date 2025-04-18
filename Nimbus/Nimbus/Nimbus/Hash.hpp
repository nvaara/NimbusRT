#pragma once
#include <Nimbus/Types.hpp>
#include <Nimbus/Constants.hpp>
#include <array>
#include <xhash>

namespace Nimbus::Utils
{
	template <typename Type, typename... Types>
	inline void CombineHash(std::size_t& seed, const Type& hashable, const Types&... hashables)
	{
		seed ^= std::hash<Type>()(hashable) + 0x9E3779B9 + (seed << 6) + (seed >> 2);
		if constexpr (sizeof...(Types) > 0)
			return CombineHash(seed, hashables...);
	}

	template <typename... Types>
	size_t CalculateHash(const Types&... types)
	{
		size_t hash = 0;
		CombineHash(hash, types...);
		return hash;
	}
}
namespace Nimbus
{
	class PathHash
	{
	public:
		PathHash(const PathInfo& pathInfo, const uint32_t* labels);
		size_t GetHash() const { return m_HashKey; }
		bool AreLabelsEqual(const PathHash& other) const;
		bool operator==(const PathHash& other) const;

	private:
		PathInfo m_PathInfo;
		std::array<uint32_t, Constants::MaximumNumberOfInteractions> m_Labels;
		size_t m_HashKey;
	};

	struct PathHashKey
	{
		PathHashKey(size_t key) : key(key) {}
		bool operator==(const PathHashKey& k) const { return k.key == key; }
		size_t key;
	};

}

template <> struct std::hash<Nimbus::PathHash>
{
	inline size_t operator()(const Nimbus::PathHash& hash) const
	{
		return hash.GetHash();
	}
};