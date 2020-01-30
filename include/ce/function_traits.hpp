#pragma once

namespace ce
{
    namespace function_traits
    {
        template <class T, class R, class... FArgs>
        constexpr bool is_const(R (T::*)(FArgs...))
        {
            return false;
        }
        template <class T, class R, class... FArgs>
        constexpr bool is_const(R (T::*)(FArgs...) const)
        {
            return true;
        }
    }
}
