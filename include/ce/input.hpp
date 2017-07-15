#pragma once

template<class T> struct HashedInput {
    template<class...Args> HashedInput(Args&&... args) :
        data(std::forward<Args>(args)...) {
        hash = generateHash();
    }

    size_t hash;
    T data;
};

template<class T, class... Args> HashedInput<T> make_input(Args&&... args) {
    return HashedInput<T>(std::forward<Args>(args)...);
}

template<class T>
std::size_t combineHash(std::size_t seed, const HashedInput<T>& v) {
    seed ^= v.hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

template<class T> T& get(HashedInput<T>& data) {
    return data.data;
}