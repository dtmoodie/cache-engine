#pragma once

namespace ce{
    Hash_t generateHash() {
        static Hash_t count = 0;
        return ++count;
    }

    template<class T>
    Hash_t generateHash(const T& data) {
        std::hash<T> hasher;
        return hasher(data);
    }

    template<class T>
    Hash_t combineHash(std::size_t seed, const T& v) {
        seed ^= generateHash(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    Hash_t combineHash(std::size_t seed, std::size_t hash) {
        seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    template<class T>
    Hash_t generateHash(std::size_t seed, T&& v) {
        return combineHash(seed, std::forward<T>(v));
    }

    Hash_t generateHash(std::size_t seed) {
        return seed;
    }

    template<class T>
    Hash_t generateHash(const std::vector<T>& data) {
        std::hash<T> hasher;
        Hash_t hash = 0;
        for (const auto& val : data) {
            hash = combineHash(hash, hasher(val));
        }
        return hash;
    }

    template<class T, class R, class... FArgs>
    Hash_t generateHash(R(T::*func)(FArgs...)) {
        std::hash<R(T::*)(FArgs...)> hasher;
        return hasher(func);
    }

    template<class R, class... FArgs>
    Hash_t generateHash(R(*func)(FArgs...)) {
        return reinterpret_cast<std::size_t>((void*)func);
    }

    template<class T, class...Args>
    Hash_t generateHash(std::size_t seed, T&& v, Args&&... args) {
        return generateHash(combineHash(seed, std::forward<T>(v)), std::forward<Args>(args)...);
    }

    template<class T> 
    constexpr const char* ClassHasher<T>::name() {
        return __FUNCTION__;
    }

    template<class T>
    constexpr uint32_t ClassHasher<T>::hash() {
        return ct::ctcrc32(__FUNCTION__);
    }

    template<class T>
    constexpr uint32_t classHash() {
        return ClassHasher<T>::hash();
    }
    template<class T>
    constexpr const char* className() {
        return ClassHasher<T>::name();
    }
}