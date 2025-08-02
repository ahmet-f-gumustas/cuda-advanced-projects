#include "cublaslt_utils.hpp"

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
    cudaStream_t stream) {
    
    // Create matrix layouts
    CublasLtMatrixLayout layoutA(CUDA_R_16F, 
        transa == CUBLAS_OP_N ? k : m, 
        transa == CUBLAS_OP_N ? m : k, lda);
    
    CublasLtMatrixLayout layoutB(CUDA_R_16F,
        transb == CUBLAS_OP_N ? n : k,
        transb == CUBLAS_OP_N ? k : n, ldb);
    
    CublasLtMatrixLayout layoutC(CUDA_R_16F, n, m, ldc);
    
    // Create matmul descriptor
    CublasLtMatmulDesc matmulDesc(CUBLAS_COMPUTE_16F, CUDA_R_16F);
    
    // Set transpose operations
    matmulDesc.set_attribute(CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    matmulDesc.set_attribute(CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    
    // Create preference descriptor
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    
    // Set workspace if provided
    if (workspace && workspace_size > 0) {
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size,
            sizeof(workspace_size)
        ));
    }
    
    // Find best algorithm
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle,
        matmulDesc,
        layoutA,
        layoutB,
        layoutC,
        layoutC,
        preference,
        1,
        &heuristicResult,
        &returnedResults
    ));
    
    if (returnedResults == 0) {
        throw std::runtime_error("No suitable algorithm found for matrix multiplication");
    }
    
    // Execute matmul
    CUBLAS_CHECK(cublasLtMatmul(
        handle,
        matmulDesc,
        alpha,
        A, layoutA,
        B, layoutB,
        beta,
        C, layoutC,
        C, layoutC,
        &heuristicResult.algo,
        workspace,
        workspace_size,
        stream
    ));
    
    // Cleanup
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
}