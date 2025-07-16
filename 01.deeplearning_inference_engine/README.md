# Deep Learning Inference Engine

Yüksek performanslı, CUDA tabanlı deep learning inference engine. Modern GPU'lar için optimize edilmiş, endüstriyel kalitede bir çözüm.

## Özellikler

### Çekirdek Özellikler
- **Tensor Core Desteği**: Ampere ve üstü GPU'lar için tensor core optimizasyonu
- **INT8 Quantization**: Model boyutunu küçültme ve performans artırma
- **Multi-Stream Execution**: Paralel çalıştırma için çoklu CUDA stream desteği
- **Memory Pool**: Akıllı bellek yönetimi ve yeniden kullanım
- **Layer Fusion**: Conv+BN+ReLU gibi yaygın pattern'ler için otomatik füzyon
- **Graph Optimization**: Constant folding, dead node elimination
- **Dynamic Batching**: Inference server senaryoları için dinamik batch oluşturma

### Desteklenen Layer'lar
- Convolution (1x1, 3x3 Winograd, Depthwise, Dilated)
- Pooling (Max, Average, Global)
- Activation (ReLU, LeakyReLU, Sigmoid, Tanh, GELU)
- BatchNorm, LayerNorm, GroupNorm
- Dense (Fully Connected)
- Attention, Multi-Head Attention
- Residual connections
- Custom layer desteği

### Optimizasyonlar
- **Winograd Convolution**: 3x3 convolution için 2.25x hızlanma
- **Im2col + GEMM**: Genel convolution implementasyonu
- **Tensor Core GEMM**: FP16/INT8 mixed precision
- **Fused Kernels**: Element-wise operasyonlar için
- **Memory Coalescing**: Global memory erişim optimizasyonu
- **Shared Memory Tiling**: Cache kullanımı maksimizasyonu

## Kurulum

### Gereksinimler
- CUDA 12.0+
- cuDNN 8.0+
- CMake 3.18+
- GCC 9+ veya Clang 10+
- OpenCV (opsiyonel, örnek uygulamalar için)

### Derleme

```bash
git clone https://github.com/yourusername/deep-inference-engine.git
cd deep-inference-engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### Test

```bash
# Unit testleri çalıştır
./test_engine

# Benchmark'ları çalıştır
./benchmark_engine

# ResNet-50 inference örneği
./resnet_inference ../examples/cat.jpg
```

## Kullanım

### Basit Örnek

```cpp
#include <deep_engine/deep_engine.h>

using namespace deep_engine;

// Model oluştur
auto graph = std::make_unique<ComputationGraph>();

// Layer'ları ekle
auto conv1 = std::make_shared<ConvolutionLayer>(3, 64, 7, 2, 3);
auto relu1 = std::make_shared<ActivationLayer>("relu");

auto conv1_id = graph->add_node("conv1", conv1);
auto relu1_id = graph->add_node("relu1", relu1);

graph->add_edge(conv1_id, relu1_id);

// Optimizasyonları uygula
LayerFusionOptimizer optimizer;
optimizer.optimize(*graph);
graph->finalize();

// Inference
ExecutionContext ctx;
Tensor input({1, 3, 224, 224});
auto output = graph->forward({input}, ctx);
```

### ONNX Model Yükleme

```cpp
auto graph = ComputationGraph::from_onnx("model.onnx");

// INT8 quantization uygula
QuantizationOptimizer quant_opt(8);
quant_opt.optimize(*graph);

graph->finalize();
```

### Custom Layer Ekleme

```cpp
class MyCustomLayer : public Layer {
public:
    Tensor forward(const Tensor& input, ExecutionContext& ctx) override {
        // Custom CUDA kernel çağır
        my_kernel<<<grid, block, 0, ctx.stream()>>>(
            input.data(), output.data(), size);
        return output;
    }
};

// Layer'ı register et
REGISTER_LAYER("MyCustom", MyCustomLayer);
```

## Performans

RTX 4070 üzerinde ResNet-50 inference (batch size=1):

| Precision | Latency (ms) | Throughput (img/s) |
|-----------|--------------|-------------------|
| FP32      | 2.1          | 476               |
| FP16      | 1.2          | 833               |
| INT8      | 0.8          | 1250              |

## API Dokümantasyonu

### Tensor API

```cpp
// Tensor oluşturma
Tensor t1({64, 3, 224, 224}, DataType::FP32);
Tensor t2 = Tensor::zeros({100, 100});
Tensor t3 = Tensor::random_uniform({64, 1000}, -1.0f, 1.0f);

// View ve slice operasyonları
auto view = t1.view({64, -1});
auto slice = t1.slice(0, 0, 32);  // İlk 32 batch

// Type conversion
auto fp16_tensor = t1.to(DataType::FP16);
auto quantized = t1.quantize_int8();
```

### Graph API

```cpp
// Graph oluşturma
ComputationGraph graph;

// Node ekleme
auto node_id = graph.add_node("layer_name", layer_ptr);

// Edge ekleme
graph.add_edge(from_id, to_id);

// Input/Output işaretleme
graph.mark_input(input_id);
graph.mark_output(output_id);

// Optimizasyon ve çalıştırma
graph.optimize(optimizer);
graph.finalize();
auto outputs = graph.forward(inputs, ctx);
```

## Gelişmiş Özellikler

### Multi-GPU Desteği

```cpp
// Model'i birden fazla GPU'ya dağıt
SubgraphExtractor extractor;
auto subgraphs = extractor.extract(graph, num_gpus);

// Her GPU için executor oluştur
std::vector<GraphExecutor> executors;
for (int i = 0; i < num_gpus; ++i) {
    cudaSetDevice(i);
    executors.emplace_back(Strategy::PIPELINE);
}
```

### Profiling

```cpp
ExecutionContext ctx;
ctx.enable_profiling(true);

// Inference çalıştır
graph.forward(input, ctx);

// Timing bilgilerini al
auto timings = ctx.get_timings();
for (const auto& [layer, time] : timings) {
    std::cout << layer << ": " << time << " ms\n";
}
```

## Katkıda Bulunma

Pull request'ler kabul edilir. Büyük değişiklikler için önce issue açarak ne değiştirmek istediğinizi tartışın.

## Lisans

Bu proje MIT lisansı altındadır. Detaylar için LICENSE dosyasına bakın.