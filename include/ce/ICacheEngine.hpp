#pragma once
#include <ce/IResult.hpp>
#include <ce/export.hpp>
#include <memory>
namespace ce {
struct CE_EXPORT ICacheEngine {
    static ICacheEngine& instance();
    static void setEngine(std::unique_ptr<ICacheEngine>&& engine, bool is_thread_local = false);
    virtual std::shared_ptr<IResult>& getCachedResult(size_t hash) = 0;
};
}