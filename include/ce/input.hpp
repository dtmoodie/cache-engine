#pragma once
#include <ce/hash.hpp>
#include <ce/output.hpp>
#include <ct/type_traits.hpp>

namespace ce
{

    template <class T>
    T& get(T& data)
    {
        return data;
    }

    template <class T>
    T&& get(T&& data)
    {
        return std::forward<T>(data);
    }

    struct HashedInputBase : HashedBase
    {
    };

    template <class T>
    struct HashedInput : HashedInputBase
    {
        template <class... Args>
        HashedInput(Args&&... args)
            : data(std::forward<Args>(args)...)
        {
            setHash(generateHash());
        }

        operator ct::remove_reference_t<T>&()
        {
            return data;
        }
        operator const ct::remove_reference_t<T>&() const
        {
            return data;
        }

        T data;
    };

    template <class T>
    struct HashedInput<T&> : HashedInputBase
    {

        HashedInput(T& ref, size_t in_hash = generateHash())
            : data(ref)
        {
            setHash(in_hash);
        }

        operator ct::remove_reference_t<T>&()
        {
            return data;
        }
        operator const ct::remove_reference_t<T>&() const
        {
            return data;
        }

        T& data;
    };

    template <class T, class... Args>
    HashedInput<T> makeInput(Args&&... args)
    {
        return HashedInput<T>(std::forward<Args>(args)...);
    }

    template <class T>
    HashedInput<T&> wrapInput(T& data)
    {
        return HashedInput<T&>(data, generateHash(data));
    }

    template <class T>
    HashedInput<ct::remove_reference_t<T>&> makeInput(HashedOutput<T>& output)
    {
        HashedInput<ct::remove_reference_t<T>&> ret(output.data);
        ret.setHash(output.hash());
        return ret;
    }

    template <class T>
    HashedInput<T&> makeInput(HashedOutput<T>&& output)
    {
        HashedInput<T&> ret(output.m_ref);
        ret.hash = output.m_hash;
        return ret;
    }

    template <class T>
    T& get(HashedInput<T>& data)
    {
        return data.data;
    }

    template <class T>
    T& get(HashedInput<T&>& data)
    {
        return data.data;
    }

    template <class T>
    T& get(HashedInput<T>&& data)
    {
        return data.data;
    }

    template <class T>
    struct EmptyInput
    {
        template <class... Args>
        EmptyInput(Args&&... args)
            : data(std::forward<Args>(args)...)
        {
        }

        operator ct::remove_reference_t<T>&()
        {
            return data;
        }
        operator const ct::remove_reference_t<T>&() const
        {
            return data;
        }
        T data;
    };

    template <class T>
    EmptyInput<T&> makeEmptyInput(T&& ref)
    {
        EmptyInput<T&> ret(std::forward<T>(ref));
        return ret;
    }

    template <class T>
    EmptyInput<T&> makeEmptyInput(T& ref)
    {
        return {ref};
    }

    template <class T>
    T& get(EmptyInput<T>&& data)
    {
        return data.data;
    }

    template <class T>
    T& get(EmptyInput<T&>& data)
    {
        return data.data;
    }

    template <class T>
    const T& get(const EmptyInput<T&>& data)
    {
        return data.data;
    }

    template <class T>
    size_t getObjectHash(const EmptyInput<T>&)
    {
        return 0;
    }

    template <class T>
    size_t generateHash(const EmptyInput<T>&)
    {
        return 0;
    }

    template <class T>
    size_t generateHash(EmptyInput<T>&&)
    {
        return 0;
    }

    template <class T>
    size_t generateHash(EmptyInput<T>&)
    {
        return 0;
    }
}
