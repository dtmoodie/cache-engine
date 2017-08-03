#pragma once
#include <ce/export.hpp>
#include <iostream>
namespace ce {

template<class T> 
struct HashedOutput {
    HashedOutput(){}
    HashedOutput(const T& val, size_t hash = 0):
    m_ref(val), m_hash(hash){
    }

    HashedOutput(T&& val, size_t hash = 0):
        m_ref(std::move(val)), m_hash(hash){}

    operator T&() {return m_ref;}
    operator const T&() const {return m_ref;}

    T m_ref;
    size_t m_hash = 0;
};

// This version is used for wrapping other objects
template<class T>
struct HashedOutput<T&> {

    HashedOutput(T& ref) :
        m_ref(ref), m_hash(m_owned_hash) {}

    HashedOutput(T& ref, size_t& hash): 
    m_ref(ref), m_hash(hash){}


    operator T&() { return m_ref; }
    operator const T&() const { return m_ref; }

    T& m_ref;
    size_t& m_hash;
    size_t m_owned_hash = 0;
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
size_t generateHash(const HashedOutput<T>& output){
    return 0;
}

template<class T>
size_t combineHash(size_t seed, const HashedOutput<T>& v) {
    (void)v;
    return seed;
}

template<class T>
size_t combineHash(size_t seed, HashedOutput<T>&& v) {
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
template<class T>
constexpr bool outputDetectorHelper(ce::HashedOutput<T>* ptr = 0) {
    return true;
}
template<class T>
constexpr bool outputDetectorHelper(T* ptr = 0) {
    return false;
}

template<class T>
constexpr bool outputDetector() {
    return outputDetectorHelper(static_cast<T*>(0));
}

} // namespace ce