#pragma once
#include <ct/StringView.hpp>
#include <ct/hash.hpp>
namespace ce
{
    size_t generateHash()
    {
        static size_t count = 0;
        return ++count;
    }

    template <class T, class E = void, int I = 10>
    struct HashSelector : HashSelector<T, E, I - 1>
    {
    };

    template <class T>
    struct HashSelector<T, void, 0>
    {
        static size_t generateHash(const T& data)
        {
            std::hash<T> hasher;
            return hasher(data);
        }
    };

    template <class T>
    struct HashSelector<T, typename std::enable_if<std::is_enum<T>::value>::type, 1>
    {
        static constexpr size_t generateHash(const T data)
        {
            return data;
        }
    };

    constexpr size_t generateHash(int data)
    {
        return static_cast<size_t>(data);
    }

    template <class T>
    size_t generateHash(const T& data)
    {
        return HashSelector<T>::generateHash(data);
    }

    template <class T>
    size_t generateHash(const T* data, size_t N)
    {
        size_t hash = 0;
        for (size_t i = 0; i < N; ++i)
            hash = combineHash(hash, generateHash(data[i]));
        return hash;
    }

    template <class T>
    size_t combineHash(size_t seed, T&& v)
    {
        seed ^= generateHash(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    size_t combineHash(size_t seed, size_t hash)
    {
        seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    template <class T, class... Args>
    size_t combineHash(size_t seed, T&& arg, Args&&... args)
    {
        seed = combineHash(seed, std::forward<T>(arg));
        seed = combineHash(seed, std::forward<Args>(args)...);
        return seed;
    }

    template <class T>
    size_t generateHash(size_t seed, T&& v)
    {
        return combineHash(seed, std::forward<T>(v));
    }

    size_t generateHash(size_t seed)
    {
        return seed;
    }

    template <class T>
    inline size_t generateHash(const std::vector<T>& data)
    {
        size_t hash = 0;
        for (const auto& val : data)
        {
            hash = combineHash(hash, generateHash(val));
        }
        return hash;
    }

    template <class T, class R, class... FArgs>
    size_t generateHash(R (T::*func)(FArgs...))
    {
        std::hash<R (T::*)(FArgs...)> hasher;
        return hasher(func);
    }

    template <class R, class... FArgs>
    size_t generateHash(R (*func)(FArgs...))
    {
        return reinterpret_cast<std::size_t>((void*)func);
    }

    template <class T, class... Args>
    size_t generateHash(size_t seed, T&& v, Args&&... args)
    {
        return generateHash(combineHash(seed, std::forward<T>(v)), std::forward<Args>(args)...);
    }

    template <class T>
    constexpr const char* ClassHasher<T>::name()
    {
        return __FUNCTION__;
    }

    template <class T>
    constexpr uint32_t ClassHasher<T>::hash()
    {
        return ct::crc32(__FUNCTION__);
    }

    template <class T>
    constexpr uint32_t classHash()
    {
        return ClassHasher<T>::hash();
    }

    template <class T>
    constexpr const char* className()
    {
        return ClassHasher<T>::name();
    }
}
