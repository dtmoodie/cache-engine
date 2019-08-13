#ifndef CT_SHARED_PTR_HPP
#define CT_SHARED_PTR_HPP

#include <memory>

namespace ce
{
    template <class T>
    struct shared_ptr;

    template <class T>
    struct shared_ptr
    {
        shared_ptr() = default;

        template <class... ARGS>
        static shared_ptr create(ARGS&&... args)
        {
            return shared_ptr(std::make_shared<T>(std::forward<ARGS>(args)...));
        }

        shared_ptr(std::shared_ptr<T> data)
            : m_data(std::move(data))
        {
        }

        shared_ptr(shared_ptr<const T> data)
            : m_data(std::move(data.m_data))
            , m_is_const(true)
        {
        }

        shared_ptr(const shared_ptr<T>& data)
            : m_data(data.m_data)
            , m_is_const(false)
        {
        }

        shared_ptr& operator=(std::shared_ptr<T> v)
        {
            m_data = std::move(v);
            return *this;
        }

        shared_ptr& operator=(const shared_ptr<const T>& v)
        {
            m_data = v.m_data;
            m_is_const = true;
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

        operator std::shared_ptr<T>() const
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

      private:
        friend shared_ptr<const T>;
        void maybeCopy()
        {
            if (m_is_const && m_data)
            {
                // We don't need to copy the data if we are the only ones holding a reference to it
                // However this appears to not account for weak_ptrs.. TODO more research needed
                if(m_data.use_count() != 1)
                {
                    m_data = std::make_shared<T>(*m_data);
                    m_is_const = false;
                }
            }
        }

        std::shared_ptr<T> m_data;
        bool m_is_const = false;
    };

    template <class T>
    struct shared_ptr<const T>
    {
        shared_ptr() = default;

        template <class... ARGS>
        static shared_ptr create(ARGS&&... args)
        {
            return shared_ptr(std::make_shared<T>(std::forward<ARGS>(args)...));
        }

        shared_ptr(const shared_ptr<T>& data)
            : m_data(data.m_data)
        {
        }

        shared_ptr(const shared_ptr<const T>& data) = default;
        shared_ptr(std::shared_ptr<T> data)
            : m_data(data)
        {
        }

        shared_ptr& operator=(std::shared_ptr<T> v)
        {
            m_data = std::move(v);
            return *this;
        }

        shared_ptr& operator=(const std::shared_ptr<const T>& v)
        {
            // shh nothing to see here
            m_data = std::shared_ptr<T>(const_cast<T*>(v.get()), [v](T*) {});
            return *this;
        }

        shared_ptr& operator=(const shared_ptr<T>& v)
        {
            m_data = v.m_data;
            return *this;
        }

        shared_ptr& operator=(const shared_ptr<const T>& v)
        {
            m_data = v.m_data;
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
                return std::make_shared<T>(*m_data);
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

      private:
        friend shared_ptr<T>;
        std::shared_ptr<T> m_data;
    };
}

#endif // CT_SHARED_PTR_HPP
