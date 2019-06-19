#pragma once
#include <cstdint>
#include <vector>
namespace ce
{
    // lolol poormans hash for now
    inline size_t generateHash();

    template <class T>
    inline size_t generateHash(const T& data);

    template <class T>
    inline size_t generateHash(const T* data, size_t N);

    template <class T>
    inline size_t combineHash(size_t seed, T&& v);

    inline size_t combineHash(size_t seed, size_t hash);

    template <class T, class... Args>
    inline size_t combineHash(size_t seed, T&& arg, Args&&... args);

    template <class T>
    inline size_t generateHash(size_t seed, T&& v);

    inline size_t generateHash(size_t seed);

    template <class T>
    inline size_t generateHash(const std::vector<T>& data);

    template <class T, class R, class... FArgs>
    inline size_t generateHash(R (T::*func)(FArgs...));

    template <class R, class... FArgs>
    inline size_t generateHash(R (*func)(FArgs...));

    template <class T, class... Args>
    inline size_t generateHash(size_t seed, T&& v, Args&&... args);

    template <class T>
    struct ClassHasher
    {
        static inline constexpr const char* name();
        static inline constexpr uint32_t hash();
    };

    template <class T>
    inline constexpr uint32_t classHash();

    template <class T>
    inline constexpr const char* className();
}
#include "detail/hash.hpp"
