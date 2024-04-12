#pragma once
#include <iostream>
#include <string>
#include <string_view>
#include <mutex>

namespace VCT
{
	class Logger
	{
	public:
		template <typename... Args>
		static void Log(const std::string_view& string, Args&&... args)
		{
			static std::mutex mutex = std::mutex();
			std::lock_guard<std::mutex> lock(mutex);
			std::printf(string.data(), std::forward<Args>(args)...);
			std::cout << '\n';
		}
	};
}
#ifdef _DEBUG
#define LOGGING_ENABLED
#endif

#ifdef LOGGING_ENABLED
#define LOG(...) VCT::Logger::Log(__VA_ARGS__)
#else
#define LOG(...)
#endif
