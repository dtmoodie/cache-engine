#pragma once
#include <ce/IResult.hpp>
#include <ce/OutputPack.hpp>
#include <ce/TResult.hpp>
#include <ce/export.hpp>
#include <ce/hash.hpp>
#include <ce/input.hpp>
#include <ce/output.hpp>

#include <ct/type_traits.hpp>
#include <ct/types/TArrayView.hpp>

#include <memory>
namespace ce
{

    template <class T, class R, class... FArgs>
    struct ConstExecutionToken;

    template <class T, class R, class... FArgs>
    struct ExecutionToken;

    template <class Token, class Executor, class T, class R, class... FArgs>
    struct ExecutorHelper
    {
        ExecutorHelper(Token&& token, Executor& executor);
        template <class... Args>
        R operator()(Args&&... args);

        Executor& m_executor;
        Token m_token;
    };

    template <class T, class Derived>
    struct ExecutorBase : public Derived
    {
        template <class... Args>
        ExecutorBase(Args&&... args);

        template <class R, class... FArgs>
        ExecutorHelper<ExecutionToken<T, R, FArgs...>, ExecutorBase<T, Derived>, T, R, FArgs...>
        exec(R (T::*func)(FArgs...));

        template <class R, class... FArgs>
        ExecutorHelper<ConstExecutionToken<T, R, FArgs...>, ExecutorBase<T, Derived>, T, R, FArgs...>
        exec(R (T::*func)(FArgs...) const);

        size_t m_hash = generateHash();
    };

    template <class T>
    struct ExecutorRef
    {
        ExecutorRef(T& obj);
        T& m_obj;
    };

    template <class T>
    struct ExecutorOwner
    {
        template <class... Args>
        ExecutorOwner(Args&&... args);

        T m_obj;
    };
    template <class T>
    T& getObjectRef(ExecutorBase<T, ExecutorRef<T>>& executor);

    template <class T>
    const T& getObjectRef(const ExecutorBase<T, ExecutorRef<T>>& executor);

    template <class T>
    T& getObjectRef(ExecutorBase<T, ExecutorOwner<T>>& executor);

    template <class T>
    const T& getObjectRef(const ExecutorBase<T, ExecutorOwner<T>>& executor);

    template <class T>
    size_t& getObjectHash(ExecutorBase<T, ExecutorRef<T>>& executor);

    template <class T>
    size_t& getObjectHash(ExecutorBase<T, ExecutorOwner<T>>& executor);

    template <class T>
    size_t getObjectHash(const ExecutorBase<T, ExecutorRef<T>>& executor);

    template <class T>
    size_t getObjectHash(const ExecutorBase<T, ExecutorOwner<T>>& executor);

    template <class R, class T, class... ARGS>
    constexpr size_t memberFunctionPointerValue(R (T::*ptr)(ARGS...))
    {
        return *ct::ptrCast<size_t>(&ptr);
    }

    template <class R, class T, class... ARGS>
    constexpr size_t memberFunctionPointerValue(R (T::*ptr)(ARGS...) const)
    {
        return *ct::ptrCast<size_t>(&ptr);
    }

    template <class R, class T>
    constexpr size_t memberFunctionPointerValue(R (T::*ptr)() const)
    {
        return *ct::ptrCast<size_t>(&ptr);
    }

    struct CE_EXPORT ICacheEngine
    {
        virtual ~ICacheEngine();
        // Static singleton stuffs
        static std::shared_ptr<ICacheEngine> instance();
        static std::shared_ptr<ICacheEngine> create();
        static void setEngine(std::shared_ptr<ICacheEngine> engine);

        // These are the interface functions that must be satisfied by the implementation
        virtual std::shared_ptr<IResult> getCachedResult(size_t fhash, size_t hash) = 0;
        virtual void pushCachedResult(std::shared_ptr<IResult>, size_t fhash, size_t arg_hash) = 0;
        virtual bool printDebug() const = 0;
        virtual bool wasCacheUsedLast() const = 0;
        virtual void setCacheWasUsed(bool) = 0;

        // This is where the magic happens, funtions templated on the function signature of that which will be invoked
        /////////////////////////////////////////////////////////////////////////////
        // Static functions
        /////////////////////////////////////////////////////////////////////////////
        // Function doesn't return

        template<class... FArgs, class ... Args>
        void exec(size_t arg_hash, void(*func)(FArgs...), Args&& ... args)
        {
            using PackType = OutputPack<void, ct::remove_reference_t<Args>...>;
            using TupleType = typename PackType::types::tuple_type;
            const auto fhash = generateHash(func);
            const auto combined_hash = combineHash(fhash, arg_hash);
            if (printDebug())
            {
                std::cout << "Hash: " << arg_hash << std::endl;
            }
            std::shared_ptr<IResult> result = getCachedResult(fhash, arg_hash);
            if (result)
            {
                std::shared_ptr<TResult<TupleType>> tresult = std::dynamic_pointer_cast<TResult<TupleType>>(result);
                if (tresult)
                {
                    if (printDebug())
                    {
                        std::cout << "Found result in cache" << std::endl;
                    }
                    setCacheWasUsed(true);
                    PackType::setOutputs(combined_hash, tresult->values, args...);
                    return;
                }
            }
            setCacheWasUsed(false);
            func(ce::get(std::forward<Args>(args))...);
            TupleType results;
            PackType::saveOutputs(combined_hash, results, args...);
            result.reset(new TResult<TupleType>(std::move(results)));
            pushCachedResult(result, fhash, arg_hash);
        }

        template <class... FArgs, class... Args>
        typename std::enable_if<OutputPack<void, ct::remove_reference_t<Args>...>::OUTPUT_COUNT != 0>::type
        exec(void (*func)(FArgs...), Args&&... args)
        {
            const size_t hash = generateHash(std::forward<Args>(args)...);
            exec(hash, func, std::forward<Args>(args)...);
        }

        // function returns

        template <class R, class... FArgs, class... Args>
        HashedOutput<R> exec(size_t arg_hash, R (*func)(FArgs...), Args&&... args)
        {
            using PackType = OutputPack<void, HashedOutput<R>, ct::remove_reference_t<Args>...>;
            using TupleType = typename PackType::types::tuple_type;
            const size_t fhash = generateHash(func);
            const size_t combined_hash = combineHash(fhash, arg_hash);
            if (printDebug())
            {
                std::cout << "Hash: " << arg_hash << std::endl;
            }
            std::shared_ptr<IResult> result = getCachedResult(fhash, arg_hash);
            if (result)
            {
                std::shared_ptr<TResult<TupleType>> tresult = std::dynamic_pointer_cast<TResult<TupleType>>(result);
                if (tresult)
                {
                    if (printDebug())
                    {
                        std::cout << "Found result in cache" << std::endl;
                    }
                    setCacheWasUsed(true);
                    HashedOutput<R> ret;
                    PackType::setOutputs(combined_hash, tresult->values, ret, args...);
                    return ret;
                }
            }
            setCacheWasUsed(false);
            R ret = func(ce::get(std::forward<Args>(args))...);
            TupleType results;
            HashedOutput<R> out(std::move(ret), arg_hash);
            PackType::saveOutputs(combined_hash, results, out, args...);
            result.reset(new TResult<TupleType>(std::move(results)));
            pushCachedResult(result, fhash, arg_hash);
            return out;
        }

        template <class R, class... FArgs, class... Args>
        HashedOutput<R> exec(R (*func)(FArgs...), Args&&... args)
        {
            const size_t hash = generateHash(std::forward<Args>(args)...);
            return exec(hash, func, std::forward<Args>(args)...);
        }

        ///////////////////////////////////////////////////////////////////////////
        /// Member functions
        ///////////////////////////////////////////////////////////////////////////


        // This is the case where this is a const function with a return
        template <class T, class U, class R, class... FARGS, class... ARGS>
        HashedOutput<R> exec(size_t arg_hash, R (T::*func)(FARGS...) const, const U& obj, ARGS&&... args)
        {
            using PackType = OutputPack<void, HashedOutput<R>, ct::remove_reference_t<ARGS>...>;
            using TupleType = typename PackType::types::tuple_type;

            const auto& obj_ref = getObjectRef(obj);
            const auto fhash = memberFunctionPointerValue(func);
            const auto combined_hash = combineHash(fhash, arg_hash);
            std::shared_ptr<IResult> result = getCachedResult(fhash, arg_hash);
            if (result)
            {
                std::shared_ptr<TResult<TupleType>> tresult = std::dynamic_pointer_cast<TResult<TupleType>>(result);
                if (tresult)
                {
                    HashedOutput<R> ret;
                    if (printDebug())
                    {
                        std::cout << "Found result in cache" << std::endl;
                    }
                    setCacheWasUsed(true);
                    PackType::setOutputs(combined_hash, tresult->values, ret, args...);
                    return ret;
                }
            }
            setCacheWasUsed(false);
            TupleType results;
            HashedOutput<R> out((obj_ref.*func)(ce::get(std::forward<ARGS>(args))...), combined_hash);
            PackType::saveOutputs(combined_hash, results, out, args...);
            result.reset(new TResult<TupleType>(std::move(results)));
            pushCachedResult(result, fhash, arg_hash);
            return out;
        }

        template <class T, class U, class R, class... FARGS, class... ARGS>
        HashedOutput<R> exec(R (T::*func)(FARGS...) const, const U& obj, ARGS&&... args)
        {
            auto obj_hash = getObjectHash(obj);
            const auto arg_hash = generateHash(obj_hash, args...);
            return exec(arg_hash, func, obj, std::forward<ARGS>(args)...);
        }

        template <class T, class U, class R, class... FARGS, class... ARGS>
        HashedOutput<R> exec(size_t arg_hash, R (T::*func)(FARGS...), EmptyInput<U>& obj, ARGS&&... args)
        {
            using PackType = OutputPack<void, HashedOutput<R>, ct::remove_reference_t<ARGS>...>;
            using TupleType = typename PackType::types::tuple_type;

            auto& obj_ref = getObjectRef(obj);
            const auto fhash = memberFunctionPointerValue(func);
            const auto combined_hash = combineHash(fhash, arg_hash);
            std::shared_ptr<IResult> result = getCachedResult(fhash, arg_hash);
            if (result)
            {
                std::shared_ptr<TResult<TupleType>> tresult = std::dynamic_pointer_cast<TResult<TupleType>>(result);
                if (tresult)
                {
                    HashedOutput<R> ret;
                    if (printDebug())
                    {
                        std::cout << "Found result in cache" << std::endl;
                    }
                    setCacheWasUsed(true);
                    PackType::setOutputs(combined_hash, tresult->values, ret, args...);
                    return ret;
                }
            }
            setCacheWasUsed(false);
            TupleType results;
            HashedOutput<R> out((obj_ref.*func)(ce::get(std::forward<ARGS>(args))...), combined_hash);
            PackType::saveOutputs(combined_hash, results, out, args...);
            result.reset(new TResult<TupleType>(std::move(results)));
            pushCachedResult(result, fhash, arg_hash);
            return out;
        }

        template <class T, class U, class R, class... FARGS, class... ARGS>
        HashedOutput<R> exec(R (T::*func)(FARGS...), EmptyInput<U>&& obj, ARGS&&... args)
        {
            auto obj_hash = getObjectHash(obj);
            const auto arg_hash = generateHash(obj_hash, args...);
            return exec(arg_hash, func, obj, std::forward<ARGS>(args)...);
        }

        // Const function without a return, ie return is passed in as an argument
        template <class T, class U, class... FARGS, class... ARGS>
        void exec(size_t arg_hash, void (T::*func)(FARGS...) const, const U& object, ARGS&&... args)
        {
            using PackType = OutputPack<void, ct::remove_reference_t<ARGS>...>;
            using TupleType = typename PackType::types::tuple_type;

            static_assert(
                PackType::OUTPUT_COUNT != 0,
                "for a void returning const function, there must be some kind of output passed in as an argument");

            const auto& obj = getObjectRef(object);
            const auto fhash = memberFunctionPointerValue(func);
            const auto combined_hash = combineHash(fhash, arg_hash);

            if (printDebug())
            {
                std::cout << "Hash: " << arg_hash << std::endl;
            }
            std::shared_ptr<IResult> result = getCachedResult(fhash, arg_hash);
            if (result)
            {
                std::shared_ptr<TResult<TupleType>> tresult = std::dynamic_pointer_cast<TResult<TupleType>>(result);
                if (tresult)
                {
                    if (printDebug())
                    {
                        std::cout << "Found result in cache" << std::endl;
                    }
                    setCacheWasUsed(true);
                    PackType::setOutputs(combined_hash, tresult->values, args...);
                    return;
                }
            }

            setCacheWasUsed(false);
            (obj.*func)(ce::get(std::forward<ARGS>(args))...);
            auto results = std::make_shared<TResult<TupleType>>();
            // TupleType results;
            PackType::saveOutputs(combined_hash, results->values, args...);
            result = results;
            pushCachedResult(result, fhash, arg_hash);
        }

        template <class T, class U, class... FARGS, class... ARGS>
        void exec(void (T::*func)(FARGS...) const, const U& object, ARGS&&... args)
        {
            size_t obj_hash = getObjectHash(object);
            size_t arg_hash = generateHash(obj_hash, args...);
            exec(arg_hash, func, object, std::forward<ARGS>(args)...);
        }
    };

    template <class T>
    T& getObjectRef(ExecutorBase<T, ExecutorRef<T>>& executor)
    {
        return executor.m_obj;
    }

    template <class T>
    const T& getObjectRef(const ExecutorBase<T, ExecutorRef<T>>& executor)
    {
        return executor.m_obj;
    }

    template <class T>
    T& getObjectRef(ExecutorBase<T, ExecutorOwner<T>>& executor)
    {
        return executor.m_obj;
    }

    template <class T>
    const T& getObjectRef(const ExecutorBase<T, ExecutorOwner<T>>& executor)
    {
        return executor.m_obj;
    }

    template <class T>
    size_t& getObjectHash(ExecutorBase<T, ExecutorRef<T>>& executor)
    {
        return executor.m_hash;
    }

    template <class T>
    size_t& getObjectHash(ExecutorBase<T, ExecutorOwner<T>>& executor)
    {
        return executor.m_hash;
    }

    template <class T>
    size_t getObjectHash(const ExecutorBase<T, ExecutorRef<T>>& executor)
    {
        return executor.m_hash;
    }

    template <class T>
    size_t getObjectHash(const ExecutorBase<T, ExecutorOwner<T>>& executor)
    {
        return executor.m_hash;
    }
}
