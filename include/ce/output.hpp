#pragma once

namespace ce {

template<class T> 
struct HashedOutput {
    HashedOutput(HashedOutput<T>&& other) :
        m_ref(other.m_ref) {

    }

    HashedOutput(HashedOutput<T>& other) :
        m_ref(other.m_ref) {

    }

    HashedOutput(T& ref) :
        m_ref(ref) {}

    T& m_ref;
};

template<class T> 
HashedOutput<T> make_output(T& ref) {
    return HashedOutput<T>(ref);
}

template<class T>
std::size_t combineHash(std::size_t seed, const HashedOutput<T>& v) {
    (void)v;
    return seed;
}

template<class T> 
T& get(HashedOutput<T>&& data) {
    return data.m_ref;
}

template<class T> 
T& get(HashedOutput<T>& data) {
    return data.m_ref;
}

} // namespace ce