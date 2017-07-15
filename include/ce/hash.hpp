#pragma once

// lolol poormans hash for now
size_t generateHash() {
    static size_t count = 0;
    return ++count;
}

template<class T>
std::size_t combineHash(std::size_t seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

std::size_t combineHash(std::size_t seed, std::size_t hash) {
    seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

template<class T>
size_t generateHash(std::size_t seed, T&& v) {
    return combineHash(seed, std::forward<T>(v));
}
std::size_t generateHash(std::size_t seed) {
    return seed;
}

template<class T, class R, class... FArgs>
std::size_t generateHash(R(T::*func)(FArgs...)) {
    //return reinterpret_cast<std::size_t>((void*)func);
    std::hash<R(T::*)(FArgs...)> hasher;
    return hasher(func);
}

template<class T, class...Args>
size_t generateHash(std::size_t seed, T&& v, Args&&... args) {
    return generateHash(combineHash(seed, std::forward<T>(v)), std::forward<Args>(args)...);
}