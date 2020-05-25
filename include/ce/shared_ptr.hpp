#ifndef CT_SHARED_PTR_HPP
#define CT_SHARED_PTR_HPP

#include <ct/type_traits.hpp>

#include <functional>
#include <memory>

namespace ce
{
    template <class T, class E = void>
    struct HasClone : std::false_type
    {
    };

    template <class T>
    struct HasClone<T, ct::Valid<decltype(std::declval<T>().clone())>> : std::true_type
    {
    };

    template <class T, class U = T, class E = void>
    struct Copier
    {
        static std::shared_ptr<T> copy(const T& obj)
        {
            return std::make_shared<U>(static_cast<const U&>(obj));
        }
    };

    template <class T, class U>
    struct Copier<T, U, ct::Valid<decltype(std::declval<T>().clone())>>
    {
        static std::shared_ptr<T> copy(const T& obj)
        {
            return std::static_pointer_cast<T>(obj.clone());
        }
    };

    template <class T>
    struct shared_ptr;

    template <class T, class U = T>
    auto defaultCopyFunc(const T* ptr) -> decltype(Copier<T, U>::copy(*ptr))
    {
        return Copier<T, U>::copy(*ptr);
    }

    template <class T>
    struct shared_ptr
    {
        using element_type = T;
        using CopyFunc_t = std::function<std::shared_ptr<T>(const T*)>;

        shared_ptr() = default;
        shared_ptr(const shared_ptr&) = default;
        shared_ptr(shared_ptr&&) = default;

        template <class... ARGS>
        static shared_ptr create(ARGS&&... args)
        {
            return shared_ptr(std::make_shared<T>(std::forward<ARGS>(args)...));
        }

        template <class U>
        shared_ptr(std::shared_ptr<U> data,
                   CopyFunc_t copy_func = &defaultCopyFunc<T, U>,
                   typename std::enable_if<!std::is_const<U>::value, int>::type = 0)
            : m_data(std::move(data))
            , m_copy_func(copy_func)
        {
        }

        template <class U>
        shared_ptr(std::shared_ptr<const U> data, CopyFunc_t copy_func = &defaultCopyFunc<T, U>)
            : m_data(std::const_pointer_cast<U>(data))
            , m_copy_func(copy_func)
            , m_is_const(true)
        {
        }

        shared_ptr& operator=(std::shared_ptr<T> v)
        {
            m_data = std::move(v);
            m_copy_func = &defaultCopyFunc<T, T>;
            return *this;
        }

        shared_ptr& operator=(const shared_ptr<const T>& v)
        {
            m_data = v.m_data;
            m_copy_func = v.m_copy_func;
            m_is_const = true;
            return *this;
        }

        shared_ptr& operator=(const shared_ptr<T>& v) = default;

        const T* get() const
        {
            return m_data.get();
        }

        const T* operator->() const
        {
            return get();
        }

        const T& operator*() const
        {
            return *m_data;
        }

        std::shared_ptr<const T> data() const
        {
            return m_data;
        }

        T* get()
        {
            maybeCopy();
            return m_data.get();
        }

        T* operator->()
        {
            return get();
        }

        T& operator*()
        {
            maybeCopy();
            return *m_data;
        }

        std::shared_ptr<T> data()
        {
            return m_data;
        }

        operator bool() const
        {
            return m_data != nullptr;
        }

        operator std::shared_ptr<const T>() const
        {
            return m_data;
        }

        operator std::shared_ptr<T>()
        {
            maybeCopy();
            return m_data;
        }

        bool isConst() const
        {
            return m_is_const;
        }

        void setConst()
        {
            m_is_const = true;
        }

        template <class U>
        operator shared_ptr<U>() const
        {
            auto cpy_func = this->m_copy_func;
            auto copy_func = [cpy_func](const U* ptr) -> std::shared_ptr<U> {
                return cpy_func(static_cast<const T*>(ptr));
            };
            return shared_ptr<U>(std::shared_ptr<U>(m_data), std::move(copy_func));
        }

        template <class U>
        operator shared_ptr<const U>() const
        {
            auto cpy_func = this->m_copy_func;
            auto copy_func = [cpy_func](const U* ptr) -> std::shared_ptr<U> {
                return cpy_func(static_cast<const T*>(ptr));
            };
            return shared_ptr<const U>(std::shared_ptr<const U>(m_data), std::move(copy_func));
        }

        std::function<std::shared_ptr<T>(const T*)> getCopier() const
        {
            return m_copy_func;
        }

      private:
        friend shared_ptr<const T>;
        void maybeCopy()
        {
            if (m_is_const && m_data)
            {
                // We don't need to copy the data if we are the only ones holding a reference to it
                // However this appears to not account for weak_ptrs.. TODO more research needed
                if (m_data.use_count() != 1)
                {
                    m_data = m_copy_func(m_data.get());
                    m_is_const = false;
                }
            }
        }

        std::shared_ptr<T> m_data;
        bool m_is_const = false;
        CopyFunc_t m_copy_func;
    };

    template <class T>
    struct shared_ptr<const T>
    {
        using element_type = T;
        using CopyFunc_t = std::function<std::shared_ptr<T>(const T*)>;
        shared_ptr() = default;

        template <class... ARGS>
        static shared_ptr create(ARGS&&... args)
        {
            return shared_ptr(std::make_shared<T>(std::forward<ARGS>(args)...));
        }

        shared_ptr(const shared_ptr<T>& data)
            : m_data(data.m_data)
            , m_copy_func(data.m_copy_func)
        {
        }

        shared_ptr(const shared_ptr<const T>& data) = default;

        template <class U>
        shared_ptr(std::shared_ptr<U> data, CopyFunc_t copy_func = &defaultCopyFunc<T, U>)
            : m_data(data)
            , m_copy_func(copy_func)
        {
        }

        shared_ptr& operator=(std::shared_ptr<T> v)
        {
            m_data = std::move(v);
            m_copy_func = &defaultCopyFunc<T, T>;
            return *this;
        }

        shared_ptr& operator=(const std::shared_ptr<const T>& v)
        {
            // shh nothing to see here
            m_data = std::shared_ptr<T>(const_cast<T*>(v.get()), [v](T*) {});
            m_copy_func = &defaultCopyFunc<T, T>;
            return *this;
        }

        shared_ptr& operator=(const shared_ptr<T>& v)
        {
            m_data = v.m_data;
            m_copy_func = v.m_copy_func;
            return *this;
        }

        shared_ptr& operator=(const shared_ptr<const T>& v)
        {
            m_data = v.m_data;
            m_copy_func = v.m_copy_func;
            return *this;
        }

        const T* get() const
        {
            return m_data.get();
        }

        const T* operator->() const
        {
            return get();
        }

        const T& operator*() const
        {
            return *get();
        }

        std::shared_ptr<const T> data() const
        {
            return m_data;
        }

        operator bool() const
        {
            return m_data != nullptr;
        }

        operator std::shared_ptr<T>() const
        {
            if (m_data)
            {
                return m_copy_func(m_data.get());
            }
            return {};
        }

        operator std::shared_ptr<const T>() const
        {
            return m_data;
        }

        bool isConst() const
        {
            return true;
        }

        void setConst()
        {
        }

        template <class U>
        operator shared_ptr<U>() const
        {
            auto cpy_func = this->m_copy_func;
            auto copy_func = [cpy_func](const U* ptr) -> std::shared_ptr<U> {
                return cpy_func(static_cast<const T*>(ptr));
            };
            return shared_ptr<U>(std::shared_ptr<const U>(m_data), std::move(copy_func));
        }

        CopyFunc_t getCopier() const
        {
            return m_copy_func;
        }

      private:
        friend shared_ptr<T>;
        std::shared_ptr<T> m_data;
        CopyFunc_t m_copy_func;
    };

    template <class T, class... ARGS>
    shared_ptr<T> make_shared(ARGS&&... args)
    {
        auto ptr = std::make_shared<T>(std::forward<ARGS>(args)...);
        return shared_ptr<T>(ptr);
    }
} // namespace ce

#endif // CT_SHARED_PTR_HPP
