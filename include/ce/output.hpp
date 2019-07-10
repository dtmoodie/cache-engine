#pragma once
#include <ce/export.hpp>
#include <ce/hash.hpp>

#include <iostream>
#include <memory>
namespace ce
{
    struct HashedBase
    {
        size_t hash() const;

        void setHash(size_t val);

      private:
        size_t m_hash = 0;
    };

    template <class T>
    struct HashedOutput : public HashedBase
    {
        using type = HashedOutput<T>;
        HashedOutput(T val = {})
            : m_ref(std::move(val))
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
    };

    // This version is used for wrapping other objects
    template <class T>
    struct HashedOutput<T&> : public HashedBase
    {
        using type = HashedOutput<T&>;

        HashedOutput(T& ref)
            : m_ref(ref)
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
    };

    template <class T, class E = void, int P = 10>
    struct ReturnSelector : ReturnSelector<T, E, P - 1>
    {
    };

    template <class T>
    struct ReturnSelector<T, void, 0>
    {
        using type = HashedOutput<T>;
    };

    template <>
    struct ReturnSelector<void, void, 1>
    {
        using type = void;
    };

    template <class T>
    struct ReturnSelector<T, typename std::enable_if<std::is_base_of<HashedBase, T>::value>::type, 9>
    {
        using type = T;
    };

    template <class T>
    struct ReturnSelector<
        std::shared_ptr<T>,
        typename std::enable_if<std::is_base_of<HashedBase, typename std::decay<T>::type>::value>::type,
        9>
    {
        using type = std::shared_ptr<T>;
    };

    template <class T>
    using ReturnSelect = typename ReturnSelector<T>::type;

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const HashedOutput<T>& value)
    {
        os << value.m_ref << ':' << value.hash();
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
    T& get(HashedOutput<T&> data)
    {
        return data.m_ref;
    }

    template <class T>
    struct HashSelector<T, typename std::enable_if<std::is_base_of<ce::HashedBase, T>::value>::type, 9>
    {
        static size_t generateHash(const HashedBase& v)
        {
            return v.hash();
        }
    };
} // namespace ce
