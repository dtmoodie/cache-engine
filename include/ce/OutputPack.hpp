#pragma once
#include <ce/VariadicTypedef.hpp>
#include <ce/output.hpp>
#include <ce/input.hpp>
#include <ce/type_traits.hpp>
#include <tuple>

namespace ce {
    namespace as = type_traits::argument_specializations;

    template<class FSig, class ... Args>
    struct ArgumentIterator {
        static void printTypes(){
        }
    };

    template<int Idx, class Tuple, class T>
    void setOutput(size_t hash, Tuple& result, T& arg){
    
    }
    template<int Idx, class Tuple, class T>
    void setOutput(size_t hash, Tuple& result, ce::HashedOutput<T>& arg) {
        ce::get(arg) = T(std::get<Idx>(result));
        arg.m_hash = hash;
    }
    template<int Idx, class Tuple, class T>
    void saveOutput(size_t hash, Tuple& result, T& arg){
    
    }
    template<int Idx, class Tuple, class T>
    void saveOutput(size_t hash, Tuple& result, ce::HashedOutput<T>& arg) {
        std::get<Idx>(result) = ce::get(arg);
        arg.m_hash = hash;
    }

    template<class F, class Arg>
    struct ArgumentIterator<void(F), Arg>{
        typedef as::SaveType<F, Arg, 5> SaveType;
        typedef variadic_typedef<typename SaveType::type> SavePack;
        typedef typename convert_in_tuple<SavePack>::type SaveTuple;

        enum {
            IS_OUTPUT = SaveType::IS_OUTPUT,
            OUTPUT_COUNT =  IS_OUTPUT ? 1 : 0
        };
        static void debugPrint(){
            std::cout << "FArg: " << typeid(F).name() << " Arg: " << typeid(Arg).name() << (IS_OUTPUT ? " - OUTPUT" : " - INPUT") << std::endl;
            std::cout << "  SaveType:      " << typeid(SaveType).name() << std::endl;
            std::cout << "  SaveType::type " << typeid(typename SaveType::type).name() << std::endl;
            std::cout << "  SavePack:      " << typeid(SavePack).name() << std::endl;
            std::cout << "  SaveTuple:     " << typeid(SaveTuple).name() << std::endl;
        }
        static void printTypes() {
            std::cout << typeid(F).name() << " " << typeid(Arg).name() << std::endl;
        }
        static void printOutputTypes(){
            if(IS_OUTPUT){
                std::cout << "Function output: " << typeid(F).name() << " will be saved as ";
                std::cout << typeid(typename SaveType::type).name() << " for input " << typeid(Arg).name() << std::endl;
            }else{
                std::cout << "Function input: " << typeid(F).name() << " will not be saved" << std::endl;
            }
        }
        template<class TupleType>
        static void setOutputs(size_t hash, TupleType& result, Arg& out) {
            setOutput<std::tuple_size<TupleType>::value - 1>(combineHash(hash, std::tuple_size<TupleType>::value - 1), result, out);
        }
        template<class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, Arg& out) {
            saveOutput<std::tuple_size<TupleType>::value - 1>(combineHash(hash, std::tuple_size<TupleType>::value - 1), result, out);
        }
    };

    template<class F, class ... FArgs, class Arg, class... Args>
    struct ArgumentIterator<void(F, FArgs...), Arg, Args...>{
        typedef as::SaveType<F, Arg, 5> SaveType;
        typedef ArgumentIterator<void(FArgs...), Args...> Parent;
        typedef typename append_to_tupple<typename SaveType::type, typename Parent::SavePack>::type SavePack;
        typedef typename convert_in_tuple<SavePack>::type SaveTuple;
        enum {
            IS_OUTPUT = SaveType::IS_OUTPUT,
            OUTPUT_COUNT = Parent::OUTPUT_COUNT + IS_OUTPUT ? 1 : 0
        };
        
        static void debugPrint() {
            Parent::debugPrint();
            std::cout << std::endl;
            std::cout << "FArg: " << typeid(F).name() << " Arg: " << typeid(Arg).name() << (IS_OUTPUT ? " - OUTPUT" : " - INPUT") << std::endl;
            std::cout << "  SaveType:      " << typeid(SaveType).name() << std::endl;
            std::cout << "  SaveType::type " << typeid(typename SaveType::type).name() << std::endl;
            std::cout << "  SavePack:      " << typeid(SavePack).name() << std::endl;
            std::cout << "  SaveTuple:     " << typeid(SaveTuple).name() << std::endl << std::endl;
        }
        static void printTypes(){
            std::cout << typeid(F).name() << " " << typeid(Arg).name() << std::endl;
            ArgumentIterator<void(FArgs...), Args...>::printTypes();
        }
        static void printOutputTypes() {
            if(IS_OUTPUT){
                std::cout << "Function output: " << typeid(F).name() << " will be saved as ";
                std::cout << typeid(typename SaveType::type).name() << " for input " << typeid(Arg).name() << std::endl;
            }else{
                std::cout << "Function input: " << typeid(F).name() << " will not be saved" << std::endl;
            }
            ArgumentIterator<void(FArgs...), Args...>::printOutputTypes();
        }
        template<class TupleType>
        static void setOutputs(size_t hash, TupleType& result, Arg& out, Args&... args){
            setOutput<std::tuple_size<TupleType>::value - Parent::OUTPUT_COUNT - 1>(
                combineHash(hash, std::tuple_size<TupleType>::value - Parent::OUTPUT_COUNT - 1), result, out);
            Parent::setOutputs(hash, result, args...);
        }
        template<class TupleType>
        static void saveOutputs(size_t hash, TupleType& result, Arg& out, Args&... args) {
            saveOutput<std::tuple_size<TupleType>::value - Parent::OUTPUT_COUNT - 1>(
                combineHash(hash, std::tuple_size<TupleType>::value - Parent::OUTPUT_COUNT - 1),
                result, out);
            Parent::saveOutputs(hash, result, args...);
        }
    };


    template<class FSig, class ... Args>
    struct OutputPack{};

    // Specialization for function with no arguments
    template<class R>
    struct OutputPack<R(void)>{
        typedef std::tuple<R> SaveTuple;
        enum {
            IS_OUTPUT = 0,
            OUTPUT_COUNT = 1
        };
        static void debugPrint() {
            AI::debugPrint();
        }
        static void setOutputs(size_t hash, std::tuple<R>& tuple, HashedOutput<R>& ret) {
            ce::get(ret) = std::get<0>(tuple);
            ret.m_hash = hash; 
        }
        template<class TupleType>
        static void saveOutputs(size_t hash, TupleType& tuple, HashedOutput<R>& ret) {
            std::get<0>(tuple) = ce::get(ret);
            ret.m_hash = hash;
        }
    };
    template<class R, class... FArgs, class... Args>
    struct OutputPack<R(FArgs...), Args...>{
        typedef ArgumentIterator<void(FArgs...), Args...> ArgItr;
        typedef typename ArgItr::SavePack ArgSavePack;
        typedef typename append_to_tupple<R, ArgSavePack>::type SavePack;
        typedef typename convert_in_tuple<SavePack>::type SaveTuple;
        enum {
            IS_OUTPUT = 0,
            OUTPUT_COUNT = ArgItr::OUTPUT_COUNT + 1
        };
        static void debugPrint() {
            AI::debugPrint();
        }
        template<class TupleType>
        static void setOutputs(size_t hash, TupleType& tuple, HashedOutput<R>& ret, Args&... args){
            ce::get(ret) = std::get<0>(tuple);
            ret.m_hash = combineHash(hash, std::tuple_size<TupleType>::value - 1);
            ArgItr::setOutputs(hash, tuple, args...);
        }
        template<class TupleType>
        static void saveOutputs(size_t hash, TupleType& tuple, HashedOutput<R>& ret, Args&... args) {
            std::get<0>(tuple) = ce::get(ret);
            ret.m_hash = combineHash(hash, std::tuple_size<TupleType>::value - 1);
            ArgItr::saveOutputs(hash, tuple, args...);
        }
    };

    template<class... FArgs, class... Args>
    struct OutputPack<void(FArgs...), Args...>{
        typedef ArgumentIterator<void(FArgs...), Args...> ArgItr;
        typedef typename ArgItr::SavePack ArgSavePack;
        typedef typename ArgSavePack SavePack;
        typedef typename convert_in_tuple<SavePack>::type SaveTuple;
        enum {
            IS_OUTPUT = 0,
            OUTPUT_COUNT = ArgItr::OUTPUT_COUNT
        };
        static void debugPrint(){
            ArgItr::debugPrint();
        }
        template<class TupleType>
        static void setOutputs(size_t hash, TupleType& tuple, Args&... args) {
            ArgItr::setOutputs(hash, tuple, args...);
        }
        template<class TupleType>
        static void saveOutputs(size_t hash, TupleType& tuple, Args&... args) {
            ArgItr::saveOutputs(hash, tuple, args...);
        }
    };
}