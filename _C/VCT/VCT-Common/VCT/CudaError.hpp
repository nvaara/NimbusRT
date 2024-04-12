#pragma once
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <optix.h>
#include <iostream>
#include <cassert>
#include <atomic>

#define CU_CHECK(call) \
{ \
	CUresult cudaResultVariableCheck = call;\
	if (cudaResultVariableCheck != CUresult::CUDA_SUCCESS) \
	{\
		const char* error, *errStr; \
		cuGetErrorName(cudaResultVariableCheck, &error), cuGetErrorString(cudaResultVariableCheck, &errStr); \
		std::cout << "ERROR CUDA: " << error << ", " << errStr << std::endl;\
		assert(false); \
	}\
}

#define CUDA_CHECK(call) \
{ \
	cudaError cudaResultVariableCheck = call;\
	if (cudaResultVariableCheck != cudaError::cudaSuccess) \
	{\
		std::cout << "ERROR CUDA: " << cudaGetErrorName(cudaResultVariableCheck) << ", " << cudaGetErrorString(cudaResultVariableCheck) << std::endl;\
		assert(false); \
	}\
}

#define OPTIX_CHECK(call) \
{ \
	OptixResult optixResultVariableCheck = call;\
	if (optixResultVariableCheck != OPTIX_SUCCESS) \
	{\
		std::cout << "ERROR OptiX: " << optixGetErrorName(optixResultVariableCheck) << ", " << optixGetErrorString(optixResultVariableCheck) << std::endl;\
		assert(false);\
	}\
}

#define OPTIX_CHECK_LOG(call, log) \
{ \
	OptixResult optixResultVariableCheck = call;\
	if (optixResultVariableCheck != OPTIX_SUCCESS) \
	{\
		std::cout << "ERROR OptiX: " << optixGetErrorName(optixResultVariableCheck) << ", " << optixGetErrorString(optixResultVariableCheck) << std::endl;\
		std::cout << log << std::endl;\
		assert(false);\
	}\
}
