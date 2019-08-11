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
        virtual std::shared_ptr<IResult> getCachedResult(size_t fhash, size_t hash) const = 0;
        virtual void pushCachedResult(std::shared_ptr<IResult>, size_t fhash, size_t arg_hash) = 0;
        virtual bool printDebug() const = 0;
        virtual void printDebug(bool val) = 0;
        virtual bool wasCacheUsedLast() const = 0;
        virtual void setCacheWasUsed(bool) = 0;
        virtual void clearCache() = 0;

        // This is where the magic happens, funtions templated on the function signature of that which will be invoked
        /////////////////////////////////////////////////////////////////////////////
        // Static functions
        /////////////////////////////////////////////////////////////////////////////

        template <class R, class... FArgs, class... Args>
        std::shared_ptr<OutputPack<ReturnSelect<R>, ct::VariadicTypedef<FArgs...>, ct::VariadicTypedef<Args...>>>
        getCachedResult(size_t& fhash, size_t& arg_hash, R (*func)(FArgs...), Args&&... args)
        {
            fhash = generateHash(func);
            arg_hash = generateHash(std::forward<Args>(args)...);
            std::shared_ptr<IResult> result = getCachedResult(fhash, arg_hash);
            if (result)
            {
                return std::dynamic_pointer_cast<
                    OutputPack<ReturnSelect<R>, ct::VariadicTypedef<FArgs...>, ct::VariadicTypedef<Args...>>>(result);
            }
            return {};
        }

        // Function doesn't return a value
        template <class... FArgs, class... Args>
        void exec(void (*func)(FArgs...), Args&&... args)
        {
            static_assert(OutputPack<void, ct::VariadicTypedef<FArgs...>, ct::VariadicTypedef<Args...>>::OUTPUT_COUNT !=
                              0,
                          "Ouput must be passed in");
            size_t fhash, arg_hash;
            auto tresult = getCachedResult(fhash, arg_hash, func, args...);
            if (tresult)
            {
                tresult->getOutputs(args...);
                setCacheWasUsed(true);
                return;
            }
            func(ce::get(std::forward<Args>(args))...);
            tresult = std::make_shared<typename decltype(tresult)::element_type>();
            const auto combined_hash = combineHash(fhash, arg_hash);
            tresult->setHash(combined_hash);
            tresult->saveOutputs(args...);
            pushCachedResult(tresult, fhash, arg_hash);
            setCacheWasUsed(false);
        }

        // function returns a value

        template <class R, class... FArgs, class... Args>
        ReturnSelect<R> exec(R (*func)(FArgs...), Args&&... args)
        {
            size_t fhash, arg_hash;
            auto tresult = getCachedResult(fhash, arg_hash, func, std::forward<Args>(args)...);
            ReturnSelect<R> ret;
            if (tresult)
            {
                tresult->getOutputs(ret, args...);
                setCacheWasUsed(true);
                return ret;
            }
            ret = func(ce::get(std::forward<Args>(args))...);
            tresult = std::make_shared<typename decltype(tresult)::element_type>();
            const auto combined_hash = combineHash(fhash, arg_hash);
            tresult->setHash(combined_hash);
            tresult->saveOutputs(ret, args...);
            pushCachedResult(tresult, fhash, arg_hash);
            setCacheWasUsed(false);
            return ret;
        }

        ///////////////////////////////////////////////////////////////////////////
        /// Member functions
        ///////////////////////////////////////////////////////////////////////////

        template <class T, class U, class R, class... FARGS, class... ARGS>
        std::shared_ptr<OutputPack<ReturnSelect<R>, ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>>>
        getCachedResult(size_t& fhash, size_t& arg_hash, R (T::*func)(FARGS...) const, const U& obj, ARGS&&... args)
        {
            fhash = memberFunctionPointerValue(func);
            if (this->printDebug())
            {
                arg_hash = generateHashDebug(std::cout, obj, std::forward<ARGS>(args)...);
            }
            else
            {
                arg_hash = generateHash(obj, std::forward<ARGS>(args)...);
            }

            auto result = getCachedResult(fhash, arg_hash);
            if (result)
            {
                return std::dynamic_pointer_cast<
                    OutputPack<ReturnSelect<R>, ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>>>(result);
            }
            return {};
        }

        // This is the case where this is a const function with a return
        template <class T, class U, class R, class... FARGS, class... ARGS>
        ReturnSelect<R> exec(R (T::*func)(FARGS...) const, const U& obj, ARGS&&... args)
        {
            size_t fhash, arg_hash;
            auto result = getCachedResult(fhash, arg_hash, func, obj, std::forward<ARGS>(args)...);
            ReturnSelect<R> ret;
            if (result)
            {
                result->getOutputs(ret, args...);
                setCacheWasUsed(true);
                return ret;
            }
            const auto& obj_ref = get(obj);
            ret = (obj_ref.*func)(ce::get(std::forward<ARGS>(args))...);
            result = std::make_shared<typename decltype(result)::element_type>();
            const auto combined_hash = combineHash(fhash, arg_hash);
            result->setHash(combined_hash);
            result->saveOutputs(ret, args...);
            pushCachedResult(result, fhash, arg_hash);
            setCacheWasUsed(false);
            return ret;
        }

        // Const function without a return, ie return is passed in as an argument
        template <class T, class U, class... FARGS, class... ARGS>
        void exec(void (T::*func)(FARGS...) const, const U& object, ARGS&&... args)
        {
            static_assert(
                OutputPack<void, ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>>::OUTPUT_COUNT != 0,
                "for a void returning const function, there must be some kind of output passed in as an argument");

            size_t fhash, arg_hash;
            auto result = getCachedResult(fhash, arg_hash, func, object, std::forward<ARGS>(args)...);
            if (result)
            {
                result->getOutputs(args...);
                setCacheWasUsed(true);
                return;
            }
            const auto& obj = get(object);
            (obj.*func)(ce::get(std::forward<ARGS>(args))...);
            result = std::make_shared<typename decltype(result)::element_type>();
            const auto combined_hash = combineHash(fhash, arg_hash);
            result->setHash(combined_hash);
            result->saveOutputs(args...);
            pushCachedResult(result, fhash, arg_hash);
            setCacheWasUsed(false);
        }

        template <class T, class U, class... FARGS, class... ARGS>
        std::shared_ptr<OutputPack<void, ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>>>
        getCachedResult(size_t& fhash, size_t& arghash, void (T::*func)(FARGS...), U& object, ARGS&&... args)
        {
            fhash = memberFunctionPointerValue(func);
            if (this->printDebug())
            {
                arghash = generateHashDebug(std::cout, object, std::forward<ARGS>(args)...);
            }
            else
            {
                arghash = generateHash(object, std::forward<ARGS>(args)...);
            }
            // arghash = generateHash(generateHash(object), std::forward<ARGS>(args)...);
            auto result = getCachedResult(fhash, arghash);
            if (result)
            {
                return std::dynamic_pointer_cast<
                    OutputPack<void, ct::VariadicTypedef<FARGS...>, ct::VariadicTypedef<ARGS...>>>(result);
            }
            return {};
        }

        // non const function, mutates the state of object
        template <class T, class U, class... FARGS, class... ARGS>
        void exec(void (T::*func)(FARGS...), U& object, ARGS&&... args)
        {
            size_t fhash, arg_hash;
            auto result = getCachedResult(fhash, arg_hash, func, object, std::forward<ARGS>(args)...);
            auto& obj = get(object);
            if (result)
            {
                result->getOutputs(args...);
                setCacheWasUsed(true);
                return;
            }

            (obj.*func)(ce::get(std::forward<ARGS>(args))...);
            result = std::make_shared<typename decltype(result)::element_type>();
            const auto combined_hash = combineHash(fhash, arg_hash);
            result->setHash(combined_hash);
            result->saveOutputs(args...);
            setHash(combined_hash, object);
            pushCachedResult(result, fhash, arg_hash);
            setCacheWasUsed(false);
        }
    };
}
