#pragma once
#include "CudaError.hpp"
#include <vector>

namespace VCT
{
    class DeviceBuffer
	{
	public:
		template <typename Type>
		static DeviceBuffer Create(const std::vector<Type>& data);

		DeviceBuffer();
		DeviceBuffer(CUdeviceptr devicePointer, size_t size);
		DeviceBuffer(size_t size);
		~DeviceBuffer();

		DeviceBuffer(const DeviceBuffer&) = delete;
		DeviceBuffer(DeviceBuffer&& rhs) noexcept;
		DeviceBuffer& operator=(DeviceBuffer&& rhs) noexcept;
		DeviceBuffer& operator=(const DeviceBuffer& rhs) = delete;
		operator bool() const { return m_DevicePointer > 0 && m_Size > 0; }

        template <typename Type>
		void Upload(const Type* src, size_t count, size_t first = 0) const;

		template <typename Type>
		void Download(Type* dst, size_t count, size_t first = 0) const;

		template <typename Type>
		void DownloadAsync(cudaStream_t stream, Type* dst, size_t count, size_t first = 0) const;

		CUdeviceptr GetRawHandle() const { return m_DevicePointer; }
		size_t GetSize() const { return m_Size; }

		template <typename Type>
		Type* DevicePointerCast() const;

		void Memset(int value) const;
		void MemsetZero() const;
		
	private:
		void Allocate(size_t bytes); 
		void Free();

	private:
		bool m_DestroyBuffer;
		CUdeviceptr m_DevicePointer;
		size_t m_Size;
	};

	template <typename Type>
	inline DeviceBuffer DeviceBuffer::Create(const std::vector<Type>& data)
	{
		DeviceBuffer result = DeviceBuffer(data.size() * sizeof(Type));
		result.Upload(data.data(), data.size());
		return result;
	}

    template <typename Type>
    inline void DeviceBuffer::Upload(const Type* src, size_t count, size_t first) const
    {
        CU_CHECK(cuMemcpyHtoD(m_DevicePointer + sizeof(Type) * first, src, sizeof(Type) * count));
    }

    template <typename Type>
    inline void DeviceBuffer::Download(Type* dst, size_t count, size_t first) const
    {
        CU_CHECK(cuMemcpyDtoH(dst, m_DevicePointer + sizeof(Type) * first, sizeof(Type) * count));
    }

	template <typename Type>
	inline void DeviceBuffer::DownloadAsync(cudaStream_t stream, Type* dst, size_t count, size_t first) const
	{
		CU_CHECK(cuMemcpyDtoHAsync(dst, m_DevicePointer + sizeof(Type) * first, sizeof(Type) * count, stream));
	}

	template <typename Type>
	inline Type* DeviceBuffer::DevicePointerCast() const
	{
		return reinterpret_cast<Type*>(m_DevicePointer);
	}
}