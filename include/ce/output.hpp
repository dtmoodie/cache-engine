#pragma once
#include <ce/export.hpp>
#include <ce/hash.hpp>

#include <iostream>

namespace ce
{
    template<class T>
    struct HashedBase
    {
        operator T&()
        {
            return *static_cast<T>(this);
        }

        operator const T&() const
        {
            return *static_cast<T>(this);
        }

        size_t hash() const{return m_hash;}
        void setHash(size_t val){m_hash = val;}
    private:
        size_t m_hash = 0;
    };

    template <class T>
    struct HashedOutput
    {
        HashedOutput()
        {
        }

        HashedOutput(T val, size_t hash = 0)
            : m_ref(std::move(val))
            , m_hash(hash)
        {
        }

        operator T&()
        {
            return m_ref;
        }
        operator const T&() const
        {
            return m_ref;
        }

        T m_ref;
        size_t m_hash = 0;
    };

    // This version is used for wrapping other objects
    template <class T>
    struct HashedOutput<T&>
    {

        HashedOutput(T& ref)
            : m_ref(ref)
            , m_hash(m_owned_hash)
        {
        }

        HashedOutput(T& ref, size_t& hash)
            : m_ref(ref)
            , m_hash(hash)
        {
        }

        operator T&()
        {
            return m_ref;
        }
        operator const T&() const
        {
            return m_ref;
        }

        T& m_ref;
        size_t& m_hash;
        size_t m_owned_hash = 0;
    };

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const HashedOutput<T>& value)
    {
        os << value.m_ref << ':' << value.m_hash;
        return os;
    }

    template <class T>
    HashedOutput<T&> makeOutput(T& ref)
    {
        return HashedOutput<T&>(ref);
    }

    template <class T>
    HashedOutput<T*> makeOutput(T* ptr)
    {
        return HashedOutput<T*>(ptr);
    }


    template <class T>
    T& get(HashedOutput<T>& data)
    {
        return data.m_ref;
    }


    template <class T>
    T& get(HashedBase<T>& data)
    {
        return data;
    }

    template <class T>
    struct HashSelector<HashedOutput<T>, void, 9>
    {
        static size_t generateHash(const HashedOutput<T>& data)
        {
            return data.m_hash;
        }
    };

    template<class T>
    struct HashSelector<HashedBase<T>, void, 9>
    {
        static size_t generateHash(const HashedBase<T>& v)
        {
            return v.hash();
        }
    };
} // namespace ce
