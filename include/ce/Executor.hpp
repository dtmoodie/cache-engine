#pragma once
#include <ce/output.hpp>

#include <ce/ICacheEngine.hpp>
#include <ce/OutputPack.hpp>
#include <ce/TResult.hpp>
#include <ce/input.hpp>

#include <cstring>
#include <ct/StringView.hpp>

#include <iostream>
#include <type_traits>

namespace ce
{

    template <class T>
    struct HashWrapper : public HashedBase
    {
        using value_type = T;
        HashWrapper(T v = {})
            : obj{std::move(v)}
        {
        }
        T* operator->()
        {
            return &obj;
        }

        const T* operator->() const
        {
            return &obj;
        }

        operator T&()
        {
            return obj;
        }

        T obj;
    };

    template <class T>
    struct HashWrapper<T&> : public HashedBase
    {
        using value_type = T;
        HashWrapper(T& ref)
            : obj(ref)
        {
        }
        T* operator->()
        {
            return obj;
        }

        const T* operator->() const
        {
            return obj;
        }

        operator T&()
        {
            return obj;
        }

        T& obj;
    };

    template <class T>
    struct HashWrapper<T*> : public HashedBase
    {
        using value_type = T;
        T* operator->()
        {
            return obj;
        }

        const T* operator->() const
        {
            return obj;
        }

        operator T&()
        {
            return *obj;
        }

        T* obj;
    };

    namespace result_traits
    {
        template <class T, class U>
        struct IsOutput<HashWrapper<T>, U, void, 4>
        {
            static constexpr const bool value = false;
            using result_storage_type = ct::VariadicTypedef<T>;
        };

        template <class U, class T>
        struct Storage<U, HashWrapper<T>>
        {
            using type = ct::VariadicTypedef<T>;
        };

        template <size_t IDX, class T, class TupleType, class... Args>
        void getResult(size_t hash, const TupleType& result, HashWrapper<T>& out, Args&&...)
        {
            ce::get(out) = deepCopy(std::get<IDX>(result));
            out.setHash(ct::combineHash(hash, IDX));
        }

        template <size_t IDX, class T, class TupleType, class... Args>
        void saveResult(size_t hash, TupleType& result, HashWrapper<T>& out, Args&&...)
        {
            std::get<IDX>(result) = deepCopy(ce::get(out));
            out.setHash(ct::combineHash(hash, IDX));
        }

    } // namespace result_traits

    template <class T>
    T& get(HashWrapper<T>& v)
    {
        return v.obj;
    }

    template <class T>
    const T& get(const HashWrapper<T>& v)
    {
        return v.obj;
    }

    template <class T>
    T& get(HashWrapper<T*>& v)
    {
        return *v.obj;
    }

    template <class T>
    const T& get(const HashWrapper<T*>& v)
    {
        return *v.obj;
    }

    template <class T>
    T& get(HashWrapper<T&>& v)
    {
        return v.obj;
    }

    template <class T>
    const T& get(const HashWrapper<T&>& v)
    {
        return v.obj;
    }

    template <class T>
    typename std::enable_if<!std::is_base_of<HashedBase, T>::value, HashWrapper<T>>::type wrapHash(T&& r = T{})
    {
        return std::move(r);
    }

    template <class T>
    typename std::enable_if<!std::is_base_of<HashedBase, T>::value, HashWrapper<T&>>::type wrapHash(T& r)
    {
        return HashWrapper<T&>(r);
    }

    template <class T>
    typename std::enable_if<!std::is_base_of<HashedBase, T>::value, HashWrapper<T*>>::type wrapHash(T* r)
    {
        return r;
    }

    template <class T>
    typename std::enable_if<std::is_base_of<HashedBase, T>::value, T>::type wrapHash(T r)
    {
        return r;
    }

    template <class T>
    typename std::enable_if<std::is_base_of<HashedBase, T>::value, T>::type wrapHash(T&& r)
    {
        return r;
    }
} // namespace ce
