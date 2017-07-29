#pragma once

#include <vector>
namespace ce {
// lolol poormans hash for now
inline Hash_t generateHash();

template<class T>
inline Hash_t generateHash(const T& data);

template<class T>
inline Hash_t combineHash(std::size_t seed, const T& v);

inline Hash_t combineHash(std::size_t seed, std::size_t hash);

template<class T>
inline Hash_t generateHash(std::size_t seed, T&& v);

inline Hash_t generateHash(std::size_t seed);

template<class T>
inline Hash_t generateHash(const std::vector<T>& data);

template<class T, class R, class... FArgs>
inline Hash_t generateHash(R(T::*func)(FArgs...));

template<class R, class... FArgs>
inline Hash_t generateHash(R(*func)(FArgs...));

template<class T, class...Args>
inline Hash_t generateHash(std::size_t seed, T&& v, Args&&... args);

template<class T> struct ClassHasher {
    static inline constexpr const char* name();
    static inline constexpr uint32_t hash();
};

template<class T>
inline constexpr uint32_t classHash();

template<class T>
inline constexpr const char* className();
}
#include "detail/hash.hpp"