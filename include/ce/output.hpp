#pragma once
#include <ce/export.hpp>
#include <iostream>
namespace ce {

template<class T> 
struct HashedOutput {
    HashedOutput(){}
    HashedOutput(const T& val, Hash_t hash = 0):
    m_ref(val), m_hash(hash){
    }

    HashedOutput(T&& val, Hash_t hash = 0):
        m_ref(std::move(val)), m_hash(hash){}

    operator T&() {return m_ref;}
    operator const T&() const {return m_ref;}

    T m_ref;
    Hash_t m_hash = 0;
};

// This version is used for wrapping other objects
template<class T>
struct HashedOutput<T&> {

    HashedOutput(T& ref) :
        m_ref(ref), m_hash(m_owned_hash) {}

    HashedOutput(T& ref, Hash_t& hash): 
    m_ref(ref), m_hash(hash){}


    operator T&() { return m_ref; }
    operator const T&() const { return m_ref; }

    T& m_ref;
    Hash_t& m_hash;
    Hash_t m_owned_hash = 0;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const HashedOutput<T>& value){
    os << value.m_ref << ':' << value.m_hash;
    return os;
}

template<class T> 
HashedOutput<T&> makeOutput(T& ref) {
    return HashedOutput<T&>(ref);
}

template<class T>
HashedOutput<T*> makeOutput(T* ptr) {
	return HashedOutput<T*>(ptr);
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