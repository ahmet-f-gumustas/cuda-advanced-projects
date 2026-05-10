#include "image_io.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <algorithm>

Image read_ppm(const std::string& path) {
    Image img;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return img;

    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fclose(f);
        return img;
    }

    int w = 0, h = 0, maxv = 0;
    auto skip_ws_and_comments = [&]() {
        int c = fgetc(f);
        while (c != EOF) {
            if (c == '#') {
                while (c != '\n' && c != EOF) c = fgetc(f);
            } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                c = fgetc(f);
            } else {
                ungetc(c, f);
                break;
            }
        }
    };
    skip_ws_and_comments();
    if (fscanf(f, "%d", &w) != 1) { fclose(f); return img; }
    skip_ws_and_comments();
    if (fscanf(f, "%d", &h) != 1) { fclose(f); return img; }
    skip_ws_and_comments();
    if (fscanf(f, "%d", &maxv) != 1 || maxv != 255) { fclose(f); return img; }
    // Single whitespace after header
    fgetc(f);

    img.width = w;
    img.height = h;
    img.pixels.resize((size_t)w * h * 3);
    if (fread(img.pixels.data(), 1, img.pixels.size(), f) != img.pixels.size()) {
        img = Image{};
    }
    fclose(f);
    return img;
}

bool write_ppm(const std::string& path, const Image& img) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fprintf(f, "P6\n%d %d\n255\n", img.width, img.height);
    size_t n = (size_t)img.width * img.height * 3;
    bool ok = fwrite(img.pixels.data(), 1, n, f) == n;
    fclose(f);
    return ok;
}

Image make_synthetic_image(int w, int h, unsigned seed) {
    Image img;
    img.width = w;
    img.height = h;
    img.pixels.assign((size_t)w * h * 3, 0);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> color_dist(40, 230);
    std::uniform_int_distribution<int> size_dist(40, 200);

    // Gradient background
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * 3;
            img.pixels[idx + 0] = (uint8_t)(x * 255 / w);
            img.pixels[idx + 1] = (uint8_t)(y * 255 / h);
            img.pixels[idx + 2] = 80;
        }
    }
    // A few rectangles
    std::uniform_int_distribution<int> px(0, w - 50);
    std::uniform_int_distribution<int> py(0, h - 50);
    for (int k = 0; k < 4; ++k) {
        int x0 = px(rng), y0 = py(rng);
        int rw = std::min(size_dist(rng), w - x0 - 1);
        int rh = std::min(size_dist(rng), h - y0 - 1);
        uint8_t r = color_dist(rng), g = color_dist(rng), b = color_dist(rng);
        for (int y = y0; y < y0 + rh; ++y) {
            for (int x = x0; x < x0 + rw; ++x) {
                int idx = (y * w + x) * 3;
                img.pixels[idx + 0] = r;
                img.pixels[idx + 1] = g;
                img.pixels[idx + 2] = b;
            }
        }
    }
    return img;
}

static void draw_rect(Image& img, int x0, int y0, int x1, int y1,
                      uint8_t r, uint8_t g, uint8_t b, int thickness) {
    x0 = std::max(0, std::min(x0, img.width - 1));
    x1 = std::max(0, std::min(x1, img.width - 1));
    y0 = std::max(0, std::min(y0, img.height - 1));
    y1 = std::max(0, std::min(y1, img.height - 1));
    auto put_px = [&](int x, int y) {
        if (x < 0 || x >= img.width || y < 0 || y >= img.height) return;
        int idx = (y * img.width + x) * 3;
        img.pixels[idx + 0] = r;
        img.pixels[idx + 1] = g;
        img.pixels[idx + 2] = b;
    };
    for (int t = 0; t < thickness; ++t) {
        for (int x = x0; x <= x1; ++x) {
            put_px(x, y0 + t);
            put_px(x, y1 - t);
        }
        for (int y = y0; y <= y1; ++y) {
            put_px(x0 + t, y);
            put_px(x1 - t, y);
        }
    }
}

void draw_detections(Image& img, const std::vector<Detection>& dets) {
    // Simple color palette indexed by class_id
    static const uint8_t palette[8][3] = {
        {255, 64, 64},  {64, 255, 64},  {64, 64, 255},  {255, 255, 0},
        {255, 0, 255},  {0, 255, 255},  {255, 128, 0},  {128, 0, 255}
    };
    for (const auto& d : dets) {
        int p = ((unsigned)d.class_id) & 7;
        draw_rect(img,
                  (int)d.x1, (int)d.y1, (int)d.x2, (int)d.y2,
                  palette[p][0], palette[p][1], palette[p][2], 2);
    }
}
