#pragma once
#include <ce/VariadicTypedef.hpp>
#include <ce/output.hpp>
#include <ce/input.hpp>
namespace ce {
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

    template<class R, class ... FArgs, class T, class ... Args>
    struct OutputPack< std::enable_if_t<OutputPack<void, R(FArgs...), Args...>::OUTPUT_COUNT && !outputDetector<T>()>, R(FArgs...), T, Args...> {
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

	template<class T, class R, class ... FArgs> 
    struct OutputPack<void, R(FArgs...), HashedOutput<T>> {
		enum {
			OUTPUT_COUNT = 1
		};
		typedef variadic_typedef<std::decay_t<T>> types;

		template<class TupleType>
		static void setOutputs(size_t hash, TupleType& result, HashedOutput<T>& out) {
			ce::get(out) = std::get<std::tuple_size<TupleType>::value - 1>(result);
			out.m_hash = combineHash(hash, std::tuple_size<TupleType>::value - 1);
		}
		template<class TupleType>
		static void saveOutputs(size_t hash, TupleType& result, HashedOutput<T>& out) {
			std::get<std::tuple_size<TupleType>::value - 1>(result) = ce::get(out);
			out.m_hash = combineHash(hash, std::tuple_size<TupleType>::value - 1);
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