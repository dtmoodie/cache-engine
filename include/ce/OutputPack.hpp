#pragma once
#include <ce/VariadicTypedef.hpp>
#include <ce/output.hpp>
#include <ce/input.hpp>
#include <ce/type_traits.hpp>
#include <tuple>

namespace ce {
    namespace as = type_traits::argument_specializations;

	template<class T>
    struct DefaultOutputPack{
		enum {
			OUTPUT_COUNT = 0;
		};

		template<class TupleType>
		static void setOutputs(size_t hash, TupleType& result, T& out) {
		}

		template<class TupleType>
		static void saveOutputs(size_t hash, TupleType& result, T& out) {        
		}
	};

    

    template<class T, size_t N> 
    struct OutputPackSelector{
        
    };

    template<class T>
    struct OutputPackSelector<T, 0> {
        typedef DefaultOutputPack<T> PackType;
    };

    template<class FSig, class ... Args>
    struct ArgumentIterator{
        static void printTypes(){
        }
    };

    template<class R, class F, class Arg>
    struct ArgumentIterator<R(F), Arg>{
        enum {
            OUTPUT_COUNT = (as::hasOutputSpecialization<F>() || as::hasOutputSpecialization<Arg>()) ? 1 : 0
        };
        static void printTypes() {
            std::cout << typeid(F).name() << " " << typeid(Arg).name() << std::endl;
        }
        static void printOutputTypes(){
            std::cout << typeid(as::SaveType_t<F>).name() << " " << typeid(as::SaveType_t<Arg>).name() << std::endl;
        }
    };

    template<class R, class F, class ... FArgs, class Arg, class... Args>
    struct ArgumentIterator<R(F, FArgs...), Arg, Args...>{
        enum{
            OUTPUT_COUNT = ArgumentIterator<R(FArgs...), Args...>::OUTPUT_COUNT + (as::hasOutputSpecialization<F>() || as::hasOutputSpecialization<Arg>()) ? 1 : 0
        };

        static void printTypes(){
            std::cout << typeid(F).name() << " " << typeid(Arg).name() << std::endl;
            ArgumentIterator<R(FArgs...), Args...>::printTypes();
        }
        static void printOutputTypes() {
            std::cout << typeid(as::SaveType_t<F>).name() << " " << typeid(as::SaveType_t<Arg>).name() << std::endl;
            ArgumentIterator<R(FArgs...), Args...>::printOutputTypes();
        }
    };

}