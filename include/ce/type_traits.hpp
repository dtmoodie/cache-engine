#pragma once
#include <utility>
namespace ce{
namespace type_traits{
namespace argument_specializations{

    template<class T>
    struct remove_output {
        typedef T type;
    };

    template<class T>
    struct remove_output<ce::HashedOutput<T>> {
        typedef std::remove_reference_t<T> type;
    };

    template<class FArg, class Arg, int Priority>
    struct SaveType{
        enum{IS_OUTPUT = 0};
        typedef void type;
        static size_t hash(const Arg& hash){
            return generateHash(hash);
        }
    };

    template<class FArg, class Arg>
    struct SaveType<FArg, Arg, 0> {
        enum { IS_OUTPUT = 0 };
        typedef void type;
        static size_t hash(const Arg& hash) {
            return generateHash(hash);
        }
    };

    template<class FArg, class Arg>
    struct SaveType<FArg, Arg, 5>: public SaveType<FArg, Arg, 4>{
        inline static size_t hash(const Arg& val){
            return SaveType<FArg, Arg, 4>::hash(val);
        }
    };
    template<class FArg, class Arg>
    struct SaveType<FArg, Arg, 4> : public SaveType<FArg, Arg, 3> {
        inline static size_t hash(const Arg& val) {
            return SaveType<FArg, Arg, 3>::hash(val);
        }
    };
    template<class FArg, class Arg>
    struct SaveType<FArg, Arg, 3> : public SaveType<FArg, Arg, 2> {
        inline static size_t hash(const Arg& val) {
            return SaveType<FArg, Arg, 2>::hash(val);
        }
    };
    template<class FArg, class Arg>
    struct SaveType<FArg, Arg, 2> : public SaveType<FArg, Arg, 1> {
        inline static size_t hash(const Arg& val) {
            return SaveType<FArg, Arg, 1>::hash(val);
        }
    };
    template<class FArg, class Arg>
    struct SaveType<FArg, Arg, 1> : public SaveType<FArg, Arg, 0> {
        inline static size_t hash(const Arg& val) {
            return SaveType<FArg, Arg, 0>::hash(val);
        }
    };
    template<class FArg, class Arg>
    struct SaveType<ce::HashedOutput<FArg>, Arg, 1> {
        enum { IS_OUTPUT = 1 };
        typedef FArg type;
    };

    template<class FArg, class Arg>
    struct SaveType<FArg, ce::HashedOutput<Arg>, 1> {
        enum { IS_OUTPUT = 1 };
        typedef std::remove_reference_t<FArg> type;
        static size_t hash(const ce::HashedOutput<Arg>& hash) {
            return 0;
        }
    };

    template<class Arg>
    struct SaveType<Arg&, ce::HashedOutput<Arg>, 1> {
        enum { IS_OUTPUT = 1 };
        typedef Arg type;
        static size_t hash(const ce::HashedOutput<Arg>& hash) {
            return 0;
        }
    };

    template<class FArg>
    struct SaveType<ce::HashedOutput<FArg>, ce::HashedOutput<FArg>, 1> {
        enum { IS_OUTPUT = 1 };
        typedef FArg type;
        static size_t hash(const ce::HashedOutput<FArg>& hash) {
            return 0;
        }
    };

} // namespace ce::type_traits::argument_specializations
} // namespace ce::type_traits
} // namespace ce