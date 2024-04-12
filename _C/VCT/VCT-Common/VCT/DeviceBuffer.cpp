#include "DeviceBuffer.hpp"

namespace VCT
{
    DeviceBuffer::DeviceBuffer()
        : m_DestroyBuffer(false)
        , m_DevicePointer(0)
        , m_Size(0)
    {

    }
    
    DeviceBuffer::DeviceBuffer(CUdeviceptr devicePointer, size_t size)
        : m_DestroyBuffer(false)
        , m_DevicePointer(devicePointer)
        , m_Size(size)
    {

    }

    DeviceBuffer::DeviceBuffer(size_t size)
        : m_DestroyBuffer(false)
        , m_DevicePointer(0)
        , m_Size(0)
    {
        Allocate(size);
    }

    DeviceBuffer::~DeviceBuffer()
    {
        Free();
    }

    DeviceBuffer::DeviceBuffer(DeviceBuffer&& rhs) noexcept
        : m_DestroyBuffer(rhs.m_DestroyBuffer)
        , m_DevicePointer(rhs.m_DevicePointer)
        , m_Size(rhs.m_Size)
    {
        rhs.m_DestroyBuffer = false;
        rhs.m_DevicePointer = 0;
        rhs.m_Size = 0;
    }

    DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& rhs) noexcept
    {
        if (this != &rhs)
        {
            Free();
            m_DestroyBuffer = rhs.m_DestroyBuffer;
            m_DevicePointer = rhs.m_DevicePointer;
            m_Size = rhs.m_Size;

            rhs.m_DestroyBuffer = false;
            rhs.m_DevicePointer = 0;
            rhs.m_Size = 0;
        }
        return *this;
    }

    void DeviceBuffer::Memset(int value) const
    {
        CUDA_CHECK(cudaMemset(DevicePointerCast<void>(), value, GetSize()));
    }

    void DeviceBuffer::MemsetZero() const
    {
        Memset(0);
    }

    void DeviceBuffer::Allocate(size_t bytes)
    {
        m_DestroyBuffer = true;
        m_Size = bytes;
        CU_CHECK(cuMemAlloc(&m_DevicePointer, bytes));
    }

    void DeviceBuffer::Free()
    {
        if (m_DestroyBuffer)
        {
            CU_CHECK(cuMemFree(m_DevicePointer));
            m_DestroyBuffer = false;
        }
    }
}