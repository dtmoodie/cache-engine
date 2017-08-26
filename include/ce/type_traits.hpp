#pragma once
#include <utility>
namespace ce{
namespace type_traits{
namespace argument_specializations{

    template<class FArg, class Arg>
    struct SaveType{
        enum{IS_OUTPUT = 0};
        typedef void type;
    };

    template<class FArg, class Arg>
    struct SaveType<ce::HashedOutput<FArg>, Arg> {
        enum { IS_OUTPUT = 1 };
        typedef FArg type;
    };

    template<class FArg, class Arg>
    struct SaveType<FArg, ce::HashedOutput<Arg>> {
        enum { IS_OUTPUT = 1 };
        typedef std::remove_reference_t<Arg> type;
    };

    template<class Arg>
    struct SaveType<Arg&, ce::HashedOutput<Arg>> {
        enum { IS_OUTPUT = 1 };
        typedef Arg type;
    };

    template<class FArg>
    struct SaveType<ce::HashedOutput<FArg>, ce::HashedOutput<FArg>> {
        enum { IS_OUTPUT = 1 };
        typedef FArg type;
    };

} // namespace ce::type_traits::argument_specializations
} // namespace ce::type_traits
} // namespace ce