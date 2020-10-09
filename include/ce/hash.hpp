#pragma once
#include <ct/hash.hpp>
#include <ct/types/TArrayView.hpp>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ce
{
    // lolol poormans hash for now
    inline size_t generateHash();

    template <class T>
    inline size_t generateHash(const T& data);

    inline size_t generateHash(const void*);

    template <class T>
    inline size_t generateHash(T&& data);

    template <class T>
    inline size_t generateHash(const T* data, size_t N);

    template <class T>
    inline size_t combineHash(size_t seed, T&& v);

    inline size_t combineHash(size_t seed, size_t hash);

    inline size_t combineHash(size_t seed);

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

    template <class... Args>
    inline size_t generateHash(Args&&... args);

    ///////////////////////////////////////////////////////////////////////////////////
    /// IMPLEMENTATION
    ///////////////////////////////////////////////////////////////////////////////////

    size_t generateHash()
    {
        static std::atomic<size_t> count{0};
        return ++count;
    }

    inline size_t generateHash(const void* ptr)
    {
        return reinterpret_cast<size_t>(ptr);
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
    size_t generateHash(T&& data)
    {
        return HashSelector<typename std::decay<T>::type>::generateHash(data);
    }

    template <class T>
    size_t generateHash(const T* data, size_t N)
    {
        size_t hash = 0;
        for (size_t i = 0; i < N; ++i)
            hash = combineHash(hash, generateHash(data[i]));
        return hash;
    }

    size_t combineHash(size_t seed)
    {
        return seed;
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

    template <class R, class... FArgs>
    size_t generateHash(R (*func)(FArgs...))
    {
        return *ct::ptrCast<size_t>(&func);
    }

    template <class T, class... Args>
    size_t generateHash(size_t seed, T&& v, Args&&... args)
    {
        return generateHash(combineHash(seed, std::forward<T>(v)), std::forward<Args>(args)...);
    }

    template <class... Args>
    size_t generateHash(Args&&... args)
    {
        return combineHash(generateHash(std::forward<Args>(args))...);
    }

    template <class T>
    size_t generateHashDebugImpl(std::ostream& os, size_t idx, T&& arg)
    {
        auto hash = generateHash(arg);
        os << idx << ": " << hash << " ";
        return hash;
    }

    template <class T, class... ARGS>
    size_t generateHashDebugImpl(std::ostream& os, size_t idx, T&& arg, ARGS&&... args)
    {
        auto hash = generateHash(arg);
        os << idx << ": " << hash << " ";
        return combineHash(hash, generateHashDebugImpl(os, idx + 1, std::forward<ARGS>(args)...));
    }

    template <class... ARGS>
    size_t generateHashDebug(std::ostream& os, ARGS&&... args)
    {
        return generateHashDebugImpl(os, 0, std::forward<ARGS>(args)...);
    }
} // namespace ce
