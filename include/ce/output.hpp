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

    void setHash(HashedBase&, size_t);

    struct HashedOutputBase : HashedBase
    {
    };

    template <class T>
    struct HashedOutput : HashedOutputBase
    {
        using type = T;
        HashedOutput(T val = {})
            : data(std::move(val))
        {
        }

        operator T&()
        {
            return data;
        }

        operator const T&() const
        {
            return data;
        }

        template <class U>
        operator U()
        {
            return U(data);
        }

        T data;
    };

    // This version is used for wrapping other objects
    template <class T>
    struct HashedOutput<T&> : HashedOutputBase
    {
        using type = T;
        HashedOutput(T& ref)
            : data(ref)
        {
        }

        operator T&()
        {
            return data;
        }

        operator const T&() const
        {
            return data;
        }

        template <class U>
        operator U()
        {
            return U(data);
        }

        T& data;
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
        os << value.data << ':' << value.hash();
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
        return data.data;
    }

    template <class T>
    T& get(HashedOutput<T&> data)
    {
        return data.data;
    }

    template <class T>
    T& get(HashedOutput<T*> data)
    {
        assert(data.data);
        return *data.data;
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
