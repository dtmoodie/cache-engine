#pragma once

namespace ce{
namespace function_traits{
    template<class T, class R, class ... FArgs>
    constexpr bool is_const(R(T::* func)(FArgs...)){
        return false;
    }
    template<class T, class R, class ... FArgs>
    constexpr bool is_const(R(T::* func)(FArgs...) const) {
        return true;
    }
}
}