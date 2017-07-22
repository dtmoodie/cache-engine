#pragma once
#include <iostream>
namespace ce {

template<class T> 
struct HashedOutput {
    HashedOutput(HashedOutput<T>&& other) :
        m_ref(other.m_ref) {

    }

    HashedOutput(HashedOutput<T>& other) :
        m_ref(other.m_ref) {

    }

    HashedOutput(T ref) :
        m_ref(ref) {}

    HashedOutput(T& value, size_t hash):
        m_ref(value), m_hash(hash){}
    operator T&() {return m_ref;}
    operator const T&() const {return m_ref;}
    T m_ref;
    size_t m_hash = 0;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const HashedOutput<T>& value){
    os << value.m_ref << ':' << value.m_hash;
    return os;
}

template<class T> 
HashedOutput<T&> make_output(T& ref) {
    return HashedOutput<T&>(ref);
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