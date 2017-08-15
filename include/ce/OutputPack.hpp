#pragma once
#include <ce/VariadicTypedef.hpp>
#include <ce/output.hpp>
#include <ce/input.hpp>
#include <ce/type_traits.hpp>

namespace ce {
    namespace as = type_traits::argument_specializations;

    template<class Enable, class FSig, class T, class... Args> 
    struct OutputPack{
        enum {
            OUTPUT_COUNT = OutputPack<void, FSig, Args...>::OUTPUT_COUNT
        };
        template<class TupleType>
        static void setOutputs(size_t hash, TupleType& result, Args&... out) {
            std::cout << "This should never be called" << std::endl;
        }
        template<class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, Args&... out) {
            std::cout << "This should never be called" << std::endl;
        }
    };

    template<class R, class F, class ... FArgs, class T, class ... Args>
    struct OutputPack< std::enable_if_t<(as::countOutputs<Args...>() != 0 || as::countOutputs(static_cast<R(*)(FArgs...)>(nullptr))) && 
                                         as::hasDefaultSpecialization<F, T>()>, R(F, FArgs...), T, Args...> {
        enum {
            OUTPUT_COUNT = OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT
        };
        typedef typename OutputPack<void, R(FArgs...), Args...>::types types;

        template<class TupleType>
        static void setOutputs(size_t hash, TupleType& result, T& arg, Args&... out) {
            std::cout << "This should never be called" << std::endl;
        }
        template<class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, T& arg, Args&... out) {
            std::cout << "This should never be called" << std::endl;
        }
    };

    template<class FSig, class T, class ... Args>
    struct OutputPack<std::enable_if_t<OutputPack<void, FSig, Args...>::OUTPUT_COUNT && !outputDetector<T>()>, FSig, T, Args...>{
        enum {
            OUTPUT_COUNT = OutputPack<void, FSig, Args...>::OUTPUT_COUNT
        };
        typedef typename OutputPack<void, FSig, Args...>::types types;

        template<class TupleType>
        static void setOutputs(size_t hash, TupleType& result, Args&... out) {
            std::cout << "This should never be called" << std::endl;
        }
        template<class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, Args&... out) {
            std::cout << "This should never be called" << std::endl;
        }
    };

    template<class FSig, class T> struct OutputPack<void, FSig, T> {
        enum {
            OUTPUT_COUNT = 0
        };
        template<class TupleType>
        static void setOutputs(size_t hash, TupleType& result, T& out) {
            std::cout << "This should never be called" << std::endl;
        }
        template<class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, T& out) {
            std::cout << "This should never be called" << std::endl;
        }
    };

    template<class T>
    void saveOutput(size_t hash, T& value, HashedOutput<T>& out){
        value = ce::get(out);
        out.m_hash = hash;
    }

    template<class T>
    void setOutput(size_t hash, const T& value, HashedOutput<T>& out){
        ce::get(out) = value;
        out.m_hash = hash;
    }

    template<class T>
    constexpr void* saveTypeImpl(T* value = nullptr){
        return {};
    }

    template<class T>
    constexpr T* saveTypeImpl(HashedOutput<T>* value = nullptr){
        return {};
    }

    template<class T>
    constexpr std::tuple<T>* combineTypes(T* ptr = nullptr, void* ptr2 = nullptr){
        return {};
    }
    
    template<class T>
    constexpr std::tuple<T>* combineTypes(void* ptr = nullptr, T* ptr2 = nullptr) {
        return{};
    }

    template<class T>
    constexpr std::tuple<T>* combineTypes(T* ptr1 = nullptr, T* ptr2 = nullptr){
        return {};
    }

    template<class T1, class T2>
    using CombineTypes = std::remove_pointer_t<decltype(combineTypes(static_cast<T1*>(nullptr), static_cast<T2*>(nullptr)))>;

    template< class T>
    using SaveType = std::remove_pointer_t<decltype(outputTypeImpl(static_cast<T*>(nullptr)))>;
    


	template<class T, class F, class R>
    struct OutputPack<void, R(F), T> {
		enum {
			OUTPUT_COUNT = as::hasDefaultSpecialization<T, F>() ? 0 : 1;
		};
		typedef variadic_typedef<std::decay_t<SaveType<T>>> types;

		template<class TupleType>
		static void setOutputs(size_t hash, TupleType& result, HashedOutput<T>& out) {
            setOutput(combineHash(hash, std::tuple_size<TupleType>::value - 1),
                      std::get<std::tuple_size<TupleType>::value - 1>(result),
                      out);
		}

		template<class TupleType>
		static void saveOutputs(size_t hash, TupleType& result, HashedOutput<T>& out) {
            saveOutput(combineHash(hash, std::tuple_size<TupleType>::value - 1),
                       std::get<std::tuple_size<TupleType>::value - 1>(result),
                       out);
		}
	};

	template<class T1, class T, class R, class ... FArgs, class ... Args> 
    struct OutputPack<typename std::enable_if<OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT != 0>::type, R(T1, FArgs...), HashedOutput<T>, Args...> : public OutputPack<void, R(FArgs...), Args...> {
		enum {
			OUTPUT_COUNT = OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT + 1
		};
		typedef typename append_to_tupple<std::decay_t<T>, typename OutputPack<void, R(FArgs...), Args...>::types>::type types;

		template<typename TupleType>
		static void setOutputs(size_t hash, TupleType& result, HashedOutput<T>& out, Args&... args) {
			ce::get(out) = std::get<std::tuple_size<TupleType>::value - OUTPUT_COUNT >(result);
			out.m_hash = combineHash(hash, std::tuple_size<TupleType>::value - OUTPUT_COUNT);
			OutputPack<void, R(FArgs...), Args...>::setOutputs(hash, result, args...);
		}

		template<typename TupleType>
		static void saveOutputs(size_t hash, TupleType& result, HashedOutput<T>& out, Args&... args) {
			std::get<std::tuple_size<TupleType>::value - OUTPUT_COUNT >(result) = ce::get(out);
			out.m_hash = combineHash(hash, std::tuple_size<TupleType>::value - OUTPUT_COUNT);
			OutputPack<void, R(FArgs...), Args...>::saveOutputs(hash, result, args...);
		}
	};

	template<class T1, class T, class R, class...FArgs, class ... Args>
	struct OutputPack<typename std::enable_if<!OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT>::type, R(T1, FArgs...), HashedOutput<T>, Args...> : public OutputPack<void, R(FArgs...), Args...> {
		enum {
			OUTPUT_COUNT = 1
		};
		typedef variadic_typedef<std::decay_t<T>> types;

		template<class TupleType>
		static void setOutputs(size_t hash, TupleType& result, HashedOutput<T>& out, Args&... args) {
			ce::get(out) = std::get<std::tuple_size<TupleType>::value - OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT - 1>(result);
			out.m_hash = combineHash(hash, std::tuple_size<TupleType>::value - OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT - 1);
			OutputPack<void, R(FArgs...), Args...>::setOutputs(hash, result, args...);
		}

		template<class TupleType>
		static void saveOutputs(size_t hash, TupleType& result, HashedOutput<T>& out, Args&... args) {
			std::get<std::tuple_size<TupleType>::value - OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT - 1>(result) = ce::get(out);
			out.m_hash = combineHash(hash, std::tuple_size<TupleType>::value - OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT - 1);
			OutputPack<void, R(FArgs...), Args...>::saveOutputs(hash, result, args...);
		}
	};

	template<class FT, class T, class R, class... FArgs, class ... Args>
	struct OutputPack<typename std::enable_if<OutputPack<void, Args...>::OUTPUT_COUNT != 0 && !outputDetector<T>()>::type, R(FT, FArgs...), T, Args...> : public OutputPack<void, R(FArgs...), Args...> {
		enum {
			OUTPUT_COUNT = OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT
		};
		typedef typename OutputPack<void, R(FArgs...), Args...>::types types;

		template<typename TupleType>
		static void setOutputs(size_t hash, TupleType& result, T&, Args&... args) {
			OutputPack<void, R(FArgs...), Args...>::setOutputs(hash, result, args...);
		}

		template<typename TupleType>
		static void saveOutputs(size_t hash, TupleType& result, T&, Args&... args) {
			OutputPack<void, R(FArgs...), Args...>::saveOutputs(hash, result, args...);
		}
	};

	template<class T, class R, class ... FArgs> 
    struct OutputPack<void, R(FArgs...), T> {
		enum {
			OUTPUT_COUNT = 0
		};
		template<class TupleType>
		static void setOutputs(size_t hash, TupleType& result, T& out) {

		}
		template<class TupleType>
		static void saveOutputs(size_t hash, TupleType& result, T& out) {

		}
	};

	template<class T1, class T, class R, class ... FArgs, class ... Args>
	struct OutputPack<typename std::enable_if<!OutputPack<void, Args...>::OUTPUT_COUNT && !outputDetector<T>()>::type, R(T1, FArgs...), T, Args...> : public OutputPack<void, R(FArgs...), Args...> {
		enum {
			OUTPUT_COUNT = OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT
		};
		template<class TupleType>
		static void setOutputs(size_t hash, TupleType& result, T& out, Args&... args) {
			OutputPack<void, R(FArgs...), Args...>::setOutputs(hash, result, args...);
		}
		template<class TupleType>
		static void saveOutputs(size_t hash, TupleType& result, T& out, Args&... args) {
			OutputPack<void, R(FArgs...), Args...>::saveOutputs(hash, result, args...);
		};
	};
}