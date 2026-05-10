#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <cstdint>
#include <string>
#include <vector>

#include "postprocess.h"

// Minimal PPM (P6) image I/O. Uses uint8 RGB HWC layout.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;  // size = width * height * 3
};

// Returns Image with empty pixels on failure.
Image read_ppm(const std::string& path);

// Returns false on failure.
bool write_ppm(const std::string& path, const Image& img);

// Generate a synthetic test image with colored rectangles. Useful for
// pipeline smoke-testing without external assets.
Image make_synthetic_image(int w, int h, unsigned seed);

// Draw bounding boxes (host-side) into the image. Color cycles by class_id.
void draw_detections(Image& img, const std::vector<Detection>& dets);

#endif // IMAGE_IO_H
