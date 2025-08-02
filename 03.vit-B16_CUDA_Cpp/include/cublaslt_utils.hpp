#pragma once
#include "cuda_utils.hpp"
#include <cublasLt.h>
#include <memory>

class CublasLtHandle {
private:
    cublasLtHandle_t handle_ = nullptr;
    
public:
    CublasLtHandle() {
        CUBLAS_CHECK(cublasLtCreate(&handle_));
    }
    
    ~CublasLtHandle() {
        if (handle_) {
            cublasLtDestroy(handle_);
        }
    }
    
    CublasLtHandle(const CublasLtHandle&) = delete;
    CublasLtHandle& operator=(const CublasLtHandle&) = delete;
    
    operator cublasLtHandle_t() const { return handle_; }
    cublasLtHandle_t get() const { return handle_; }
};

class CublasLtMatmulDesc {
private:
    cublasLtMatmulDesc_t desc_ = nullptr;
    
public:
    CublasLtMatmulDesc(cublasComputeType_t compute_type, 
                       cudaDataType_t scale_type) {
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc_, compute_type, scale_type));
    }
    
    ~CublasLtMatmulDesc() {
        if (desc_) {
            cublasLtMatmulDescDestroy(desc_);
        }
    }
    
    void set_attribute(cublasLtMatmulDescAttributes_t attr, const void* value, size_t size) {
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc_, attr, value, size));
    }
    
    operator cublasLtMatmulDesc_t() const { return desc_; }
};

class CublasLtMatrixLayout {
private:
    cublasLtMatrixLayout_t layout_ = nullptr;
    
public:
    CublasLtMatrixLayout(cudaDataType_t dtype, int rows, int cols, int ld) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layout_, dtype, rows, cols, ld));
    }
    
    ~CublasLtMatrixLayout() {
        if (layout_) {
            cublasLtMatrixLayoutDestroy(layout_);
        }
    }
    
    operator cublasLtMatrixLayout_t() const { return layout_; }
};

// Helper function for FP16 GEMM with tensor cores
void gemm_fp16_tensor_core(
    cublasLtHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const __half* alpha,
    const __half* A, int lda,
    const __half* B, int ldb,
    const __half* beta,
    __half* C, int ldc,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream);