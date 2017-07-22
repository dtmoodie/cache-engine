#pragma once
#include <ce/output.hpp>
namespace ce {

template<class T> T& get(T& data) {
    return data;
}

template<class T> T&& get(T&& data) {
    return std::forward<T>(data);
}

template<class T> struct HashedInput {
    template<class...Args> HashedInput(Args&&... args) :
        data(std::forward<Args>(args)...) {
        hash = generateHash();
    }
    operator T&() { return data; }
    operator const T&() const { return data; }
    size_t hash;
    T data;
};


template<class T, class... Args> HashedInput<T> make_input(Args&&... args) {
    return HashedInput<T>(std::forward<Args>(args)...);
}

template<class T> HashedInput<T&> wrap_input(T& data){
    return HashedInput<T&>(data);
}

template<class T> HashedInput<T&> make_input(HashedOutput<T>& output) {
    HashedInput<T&> ret(output.m_ref);
    ret.hash = output.m_hash;
    return ret;
}

template<class T> HashedInput<T&> make_input(HashedOutput<T>&& output){
    HashedInput<T&> ret(output.m_ref);
    ret.hash = output.m_hash;
    return ret;
}

template<class T>
std::size_t combineHash(std::size_t seed, const HashedInput<T>& v) {
    seed ^= v.hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

template<class T> T& get(HashedInput<T>& data) {
    return data.data;
}
}