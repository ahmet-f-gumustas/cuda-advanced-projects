#ifndef CAMERA_H
#define CAMERA_H

#include <cmath>
#include <cstring>
#include <cstdio>

// ============================================================
// 4x4 Matrix (column-major, OpenGL convention)
// ============================================================

struct Mat4 {
    float m[16]; // column-major

    Mat4() { identity(); }

    void identity() {
        memset(m, 0, sizeof(m));
        m[0] = m[5] = m[10] = m[15] = 1.0f;
    }

    // Access element at (row, col) — column-major: index = col*4 + row
    float& at(int row, int col) { return m[col * 4 + row]; }
    float at(int row, int col) const { return m[col * 4 + row]; }

    Mat4 operator*(const Mat4& b) const {
        Mat4 result;
        memset(result.m, 0, sizeof(result.m));
        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                for (int k = 0; k < 4; k++) {
                    result.at(row, col) += at(row, k) * b.at(k, col);
                }
            }
        }
        return result;
    }

    void print(const char* name = "") const {
        if (name[0]) printf("%s:\n", name);
        for (int r = 0; r < 4; r++) {
            printf("  [%8.4f %8.4f %8.4f %8.4f]\n",
                   at(r, 0), at(r, 1), at(r, 2), at(r, 3));
        }
    }
};

// ============================================================
// Camera Configuration
// ============================================================

struct CameraConfig {
    int width = 1280;
    int height = 720;
    float fov_y = 60.0f;    // degrees
    float near_plane = 0.1f;
    float far_plane = 100.0f;
};

// ============================================================
// Camera class — orbit-style camera
// ============================================================

class Camera {
public:
    Camera() : position_{0.0f, 0.0f, 5.0f},
               target_{0.0f, 0.0f, 0.0f},
               up_{0.0f, 1.0f, 0.0f},
               yaw_(-90.0f), pitch_(0.0f), distance_(5.0f) {}

    Camera(float3 position, float3 target, float3 up = {0.0f, 1.0f, 0.0f})
        : position_(position), target_(target), up_(up) {
        float3 diff = {position.x - target.x, position.y - target.y, position.z - target.z};
        distance_ = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        yaw_ = atan2f(diff.z, diff.x) * 180.0f / M_PI_F;
        pitch_ = asinf(diff.y / (distance_ + 1e-8f)) * 180.0f / M_PI_F;
    }

    void setConfig(const CameraConfig& cfg) { config_ = cfg; }
    const CameraConfig& getConfig() const { return config_; }

    // Camera transformations
    Mat4 getViewMatrix() const {
        float3 f = normalizeVec(float3{target_.x - position_.x,
                                       target_.y - position_.y,
                                       target_.z - position_.z});
        float3 r = normalizeVec(crossVec(f, up_));
        float3 u = crossVec(r, f);

        Mat4 view;
        view.at(0, 0) =  r.x; view.at(0, 1) =  r.y; view.at(0, 2) =  r.z;
        view.at(1, 0) =  u.x; view.at(1, 1) =  u.y; view.at(1, 2) =  u.z;
        view.at(2, 0) = -f.x; view.at(2, 1) = -f.y; view.at(2, 2) = -f.z;
        view.at(0, 3) = -(r.x * position_.x + r.y * position_.y + r.z * position_.z);
        view.at(1, 3) = -(u.x * position_.x + u.y * position_.y + u.z * position_.z);
        view.at(2, 3) =  (f.x * position_.x + f.y * position_.y + f.z * position_.z);
        view.at(3, 3) = 1.0f;
        return view;
    }

    Mat4 getProjectionMatrix() const {
        float aspect = (float)config_.width / (float)config_.height;
        float fov_rad = config_.fov_y * M_PI_F / 180.0f;
        float tan_half_fov = tanf(fov_rad * 0.5f);
        float n = config_.near_plane;
        float f = config_.far_plane;

        Mat4 proj;
        memset(proj.m, 0, sizeof(proj.m));
        proj.at(0, 0) = 1.0f / (aspect * tan_half_fov);
        proj.at(1, 1) = 1.0f / tan_half_fov;
        proj.at(2, 2) = -(f + n) / (f - n);
        proj.at(3, 2) = -1.0f;
        proj.at(2, 3) = -(2.0f * f * n) / (f - n);
        return proj;
    }

    Mat4 getViewProjectionMatrix() const {
        return getProjectionMatrix() * getViewMatrix();
    }

    // Focal lengths in pixels
    float getFocalX() const {
        float fov_rad = config_.fov_y * M_PI_F / 180.0f;
        float tan_half_fov = tanf(fov_rad * 0.5f);
        float aspect = (float)config_.width / (float)config_.height;
        return (float)config_.width / (2.0f * aspect * tan_half_fov);
    }

    float getFocalY() const {
        float fov_rad = config_.fov_y * M_PI_F / 180.0f;
        float tan_half_fov = tanf(fov_rad * 0.5f);
        return (float)config_.height / (2.0f * tan_half_fov);
    }

    // Interactive controls
    void orbit(float dx, float dy) {
        yaw_ += dx * 0.3f;
        pitch_ += dy * 0.3f;
        pitch_ = fmaxf(-89.0f, fminf(89.0f, pitch_));
        updatePosition();
    }

    void pan(float dx, float dy) {
        float3 f = normalizeVec(float3{target_.x - position_.x,
                                       target_.y - position_.y,
                                       target_.z - position_.z});
        float3 r = normalizeVec(crossVec(f, up_));
        float3 u = crossVec(r, f);
        float speed = distance_ * 0.002f;
        target_.x += (-r.x * dx + u.x * dy) * speed;
        target_.y += (-r.y * dx + u.y * dy) * speed;
        target_.z += (-r.z * dx + u.z * dy) * speed;
        updatePosition();
    }

    void zoom(float delta) {
        distance_ *= (1.0f - delta * 0.1f);
        distance_ = fmaxf(0.1f, fminf(1000.0f, distance_));
        updatePosition();
    }

    void reset() {
        position_ = {0.0f, 0.0f, 5.0f};
        target_ = {0.0f, 0.0f, 0.0f};
        up_ = {0.0f, 1.0f, 0.0f};
        yaw_ = -90.0f;
        pitch_ = 0.0f;
        distance_ = 5.0f;
    }

    // Accessors
    float3 getPosition() const { return position_; }
    float3 getTarget() const { return target_; }
    int getWidth() const { return config_.width; }
    int getHeight() const { return config_.height; }

    void setSize(int w, int h) { config_.width = w; config_.height = h; }

    void printInfo() const {
        printf("Camera:\n");
        printf("  Position: (%.2f, %.2f, %.2f)\n", position_.x, position_.y, position_.z);
        printf("  Target:   (%.2f, %.2f, %.2f)\n", target_.x, target_.y, target_.z);
        printf("  FOV:      %.1f deg\n", config_.fov_y);
        printf("  Resolution: %dx%d\n", config_.width, config_.height);
        printf("  Focal:    fx=%.1f, fy=%.1f\n", getFocalX(), getFocalY());
        printf("\n");
    }

private:
    static constexpr float M_PI_F = 3.14159265358979323846f;

    static float3 crossVec(float3 a, float3 b) {
        return {a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x};
    }

    static float3 normalizeVec(float3 v) {
        float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
        float inv = 1.0f / (len + 1e-8f);
        return {v.x * inv, v.y * inv, v.z * inv};
    }

    void updatePosition() {
        float yaw_rad = yaw_ * M_PI_F / 180.0f;
        float pitch_rad = pitch_ * M_PI_F / 180.0f;
        position_.x = target_.x + distance_ * cosf(pitch_rad) * cosf(yaw_rad);
        position_.y = target_.y + distance_ * sinf(pitch_rad);
        position_.z = target_.z + distance_ * cosf(pitch_rad) * sinf(yaw_rad);
    }

    CameraConfig config_;
    float3 position_;
    float3 target_;
    float3 up_;
    float yaw_;
    float pitch_;
    float distance_;
};

#endif // CAMERA_H
