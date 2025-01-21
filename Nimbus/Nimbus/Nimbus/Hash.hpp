#pragma once

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

#define MAKE_HASHABLE(Type, ...) \
namespace std \
{ \
    template <> struct hash<Type> \
    { \
        size_t operator()(const Type& t) const\
    { \
        size_t hash = 0; \
        CombineHash(hash, __VA_ARGS__); \
        return hash; \
    } \
    }; \
}