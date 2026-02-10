#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace pt {

struct Vec3f {
  float x, y, z;
  Vec3f() : x(0), y(0), z(0) {}
  Vec3f(float v) : x(v), y(v), z(v) {}
  Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
  Vec3f operator+(const Vec3f& b) const { return {x + b.x, y + b.y, z + b.z}; }
  Vec3f operator-(const Vec3f& b) const { return {x - b.x, y - b.y, z - b.z}; }
  Vec3f operator*(const Vec3f& b) const { return {x * b.x, y * b.y, z * b.z}; }
  Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }
  Vec3f operator/(float s) const { float inv = 1.0f / s; return {x * inv, y * inv, z * inv}; }
  Vec3f& operator+=(const Vec3f& b) { x += b.x; y += b.y; z += b.z; return *this; }
  Vec3f operator-() const { return {-x, -y, -z}; }
  float length2() const { return x * x + y * y + z * z; }
  float length() const { return std::sqrt(length2()); }
  Vec3f normalized() const { float l = length(); return l > 0 ? *this / l : Vec3f(0); }
};

inline float dot(const Vec3f& a, const Vec3f& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3f cross(const Vec3f& a, const Vec3f& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3f operator*(float s, const Vec3f& v) { return v * s; }
inline Vec3f vmin(const Vec3f& a, const Vec3f& b) {
  return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}
inline Vec3f vmax(const Vec3f& a, const Vec3f& b) {
  return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

struct Ray {
  Vec3f origin, direction;
};

struct Material {
  Vec3f albedo = {0.8f, 0.8f, 0.8f};
  float metallic = 0.0f;
  float roughness = 0.5f;
  float opacity = 1.0f;
};

struct HitRecord {
  float t = 1e30f;
  Vec3f point;
  Vec3f normal;
  const Material* material = nullptr;
  bool hit = false;
};

struct AABB {
  Vec3f lo = Vec3f(1e30f);
  Vec3f hi = Vec3f(-1e30f);

  void expand(const Vec3f& p) {
    lo = vmin(lo, p);
    hi = vmax(hi, p);
  }

  void expand(const AABB& b) {
    lo = vmin(lo, b.lo);
    hi = vmax(hi, b.hi);
  }

  Vec3f center() const { return (lo + hi) * 0.5f; }

  int longest_axis() const {
    Vec3f d = hi - lo;
    if (d.x >= d.y && d.x >= d.z) return 0;
    if (d.y >= d.z) return 1;
    return 2;
  }

  bool intersect(const Ray& ray, float tmin, float tmax) const {
    for (int a = 0; a < 3; a++) {
      float lo_a = (&lo.x)[a];
      float hi_a = (&hi.x)[a];
      float orig = (&ray.origin.x)[a];
      float dir = (&ray.direction.x)[a];
      float inv_d = 1.0f / dir;
      float t0 = (lo_a - orig) * inv_d;
      float t1 = (hi_a - orig) * inv_d;
      if (inv_d < 0.0f) std::swap(t0, t1);
      tmin = std::max(tmin, t0);
      tmax = std::min(tmax, t1);
      if (tmax < tmin) return false;
    }
    return true;
  }
};

struct Triangle {
  Vec3f v0, v1, v2;
  Vec3f n0, n1, n2;

  AABB bounds() const {
    AABB b;
    b.expand(v0);
    b.expand(v1);
    b.expand(v2);
    return b;
  }

  Vec3f centroid() const { return (v0 + v1 + v2) / 3.0f; }

  bool intersect(const Ray& ray, float tmin, float tmax, HitRecord& rec) const {
    Vec3f e1 = v1 - v0;
    Vec3f e2 = v2 - v0;
    Vec3f h = cross(ray.direction, e2);
    float a = dot(e1, h);
    if (std::fabs(a) < 1e-8f) return false;
    float f = 1.0f / a;
    Vec3f s = ray.origin - v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    Vec3f q = cross(s, e1);
    float v = f * dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = f * dot(e2, q);
    if (t < tmin || t > tmax) return false;
    if (t < rec.t) {
      rec.t = t;
      rec.point = ray.origin + ray.direction * t;
      float w = 1.0f - u - v;
      rec.normal = (n0 * w + n1 * u + n2 * v).normalized();
      rec.hit = true;
      return true;
    }
    return false;
  }
};

struct BVHNode {
  AABB box;
  int left = -1, right = -1;
  int tri_start = -1, tri_count = 0;
  bool is_leaf() const { return tri_count > 0; }
};

class BVH {
 public:
  std::vector<Triangle> triangles;
  std::vector<BVHNode> nodes;

  void build(std::vector<Triangle>& tris) {
    triangles = std::move(tris);
    if (triangles.empty()) return;
    nodes.clear();
    nodes.reserve(triangles.size() * 2);
    std::vector<int> indices(triangles.size());
    for (int i = 0; i < (int)triangles.size(); i++) indices[i] = i;
    build_recursive(indices, 0, (int)indices.size());
    // Reorder triangles by indices for cache coherence
    std::vector<Triangle> ordered(triangles.size());
    for (int i = 0; i < (int)indices.size(); i++) ordered[i] = triangles[indices[i]];
    triangles = std::move(ordered);
  }

  bool intersect(const Ray& ray, float tmin, float tmax, HitRecord& rec) const {
    if (nodes.empty()) return false;
    return intersect_node(0, ray, tmin, tmax, rec);
  }

 private:
  int build_recursive(std::vector<int>& indices, int start, int end) {
    BVHNode node;
    AABB box;
    for (int i = start; i < end; i++) box.expand(triangles[indices[i]].bounds());
    node.box = box;

    int count = end - start;
    if (count <= 4) {
      node.tri_start = start;
      node.tri_count = count;
      int idx = (int)nodes.size();
      nodes.push_back(node);
      return idx;
    }

    int axis = box.longest_axis();
    int mid = (start + end) / 2;
    std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end,
                     [&](int a, int b) {
                       Vec3f ca = triangles[a].centroid();
                       Vec3f cb = triangles[b].centroid();
                       return (&ca.x)[axis] < (&cb.x)[axis];
                     });

    int idx = (int)nodes.size();
    nodes.push_back(node);
    nodes[idx].left = build_recursive(indices, start, mid);
    nodes[idx].right = build_recursive(indices, mid, end);
    return idx;
  }

  bool intersect_node(int node_idx, const Ray& ray, float tmin, float tmax,
                      HitRecord& rec) const {
    const BVHNode& node = nodes[node_idx];
    if (!node.box.intersect(ray, tmin, tmax)) return false;

    if (node.is_leaf()) {
      bool any_hit = false;
      for (int i = node.tri_start; i < node.tri_start + node.tri_count; i++) {
        if (triangles[i].intersect(ray, tmin, rec.t, rec)) {
          any_hit = true;
        }
      }
      return any_hit;
    }

    bool hit_left = intersect_node(node.left, ray, tmin, tmax, rec);
    bool hit_right = intersect_node(node.right, ray, tmin, rec.t, rec);
    return hit_left || hit_right;
  }
};

// PBR BRDF helpers
namespace brdf {

inline float clamp01(float x) { return std::max(0.0f, std::min(1.0f, x)); }

// GGX/Trowbridge-Reitz normal distribution
inline float distribution_ggx(float NdotH, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float d = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
  return a2 / (3.14159265f * d * d + 1e-7f);
}

// Fresnel-Schlick
inline Vec3f fresnel_schlick(float cosTheta, const Vec3f& F0) {
  float t = 1.0f - cosTheta;
  float t2 = t * t;
  float t5 = t2 * t2 * t;
  return F0 + (Vec3f(1.0f) - F0) * t5;
}

// Smith geometry (GGX)
inline float geometry_schlick_ggx(float NdotV, float roughness) {
  float r = roughness + 1.0f;
  float k = (r * r) / 8.0f;
  return NdotV / (NdotV * (1.0f - k) + k + 1e-7f);
}

inline float geometry_smith(float NdotV, float NdotL, float roughness) {
  return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

}  // namespace brdf

class PathTracer {
 public:
  void set_mesh(const std::vector<float>& positions, const std::vector<float>& normals,
                const std::vector<int>& indices, const Material& mat) {
    material_ = mat;
    std::vector<Triangle> tris;
    for (int i = 0; i < (int)indices.size(); i += 3) {
      Triangle tri;
      int i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
      tri.v0 = {positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2]};
      tri.v1 = {positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2]};
      tri.v2 = {positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2]};
      if (!normals.empty()) {
        tri.n0 = {normals[i0 * 3], normals[i0 * 3 + 1], normals[i0 * 3 + 2]};
        tri.n1 = {normals[i1 * 3], normals[i1 * 3 + 1], normals[i1 * 3 + 2]};
        tri.n2 = {normals[i2 * 3], normals[i2 * 3 + 1], normals[i2 * 3 + 2]};
      } else {
        Vec3f face_n = cross(tri.v1 - tri.v0, tri.v2 - tri.v0).normalized();
        tri.n0 = tri.n1 = tri.n2 = face_n;
      }
      tris.push_back(tri);
    }
    bvh_.build(tris);
  }

  void set_camera(const Vec3f& eye, const Vec3f& target, const Vec3f& up, float fov_deg) {
    eye_ = eye;
    target_ = target;
    up_ = up;
    fov_ = fov_deg;
  }

  void set_config(int w, int h, int max_bounces, int spp) {
    width_ = w;
    height_ = h;
    max_bounces_ = max_bounces;
    spp_ = spp;
  }

  void set_background(float r, float g, float b) { background_ = {r, g, b}; }

  const BVH& bvh() const { return bvh_; }

  void render(std::vector<uint8_t>& pixels) {
    pixels.resize(width_ * height_ * 4);

    Vec3f forward = (target_ - eye_).normalized();
    Vec3f right = cross(forward, up_).normalized();
    Vec3f cam_up = cross(right, forward).normalized();
    float aspect = (float)width_ / (float)height_;
    float half_h = std::tan(fov_ * 3.14159265f / 360.0f);
    float half_w = half_h * aspect;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int y = 0; y < height_; y++) {
      for (int x = 0; x < width_; x++) {
        Vec3f color(0.0f);
        for (int s = 0; s < spp_; s++) {
          float u = ((float)x + dist(rng)) / (float)width_;
          float v = ((float)y + dist(rng)) / (float)height_;
          float sx = (2.0f * u - 1.0f) * half_w;
          float sy = (1.0f - 2.0f * v) * half_h;
          Vec3f dir = (forward + right * sx + cam_up * sy).normalized();
          Ray ray{eye_, dir};
          color += trace(ray, 0, rng, dist);
        }
        color = color / (float)spp_;

        // Reinhard tone mapping
        color = {color.x / (color.x + 1.0f), color.y / (color.y + 1.0f),
                 color.z / (color.z + 1.0f)};

        // Gamma correction
        color = {std::pow(color.x, 1.0f / 2.2f), std::pow(color.y, 1.0f / 2.2f),
                 std::pow(color.z, 1.0f / 2.2f)};

        int idx = (y * width_ + x) * 4;
        pixels[idx + 0] = (uint8_t)(brdf::clamp01(color.x) * 255.0f);
        pixels[idx + 1] = (uint8_t)(brdf::clamp01(color.y) * 255.0f);
        pixels[idx + 2] = (uint8_t)(brdf::clamp01(color.z) * 255.0f);
        pixels[idx + 3] = 255;
      }
    }
  }

 private:
  Vec3f trace(const Ray& ray, int depth, std::mt19937& rng,
              std::uniform_real_distribution<float>& dist) {
    if (depth >= max_bounces_) return Vec3f(0.0f);

    HitRecord rec;
    if (!bvh_.intersect(ray, 0.001f, 1e30f, rec)) {
      return background_;
    }

    rec.material = &material_;
    Vec3f N = rec.normal;
    if (dot(N, ray.direction) > 0) N = -N;

    Vec3f V = -ray.direction;
    float NdotV = std::max(dot(N, V), 0.001f);

    // Sample new direction: cosine-weighted hemisphere
    Vec3f new_dir = sample_hemisphere(N, rng, dist);
    float NdotL = std::max(dot(N, new_dir), 0.0f);
    if (NdotL < 1e-6f) return Vec3f(0.0f);

    Vec3f H = (V + new_dir).normalized();
    float NdotH = std::max(dot(N, H), 0.0f);
    float VdotH = std::max(dot(V, H), 0.0f);

    const Material& mat = *rec.material;
    Vec3f F0 = Vec3f(0.04f);
    F0 = F0 * (1.0f - mat.metallic) + mat.albedo * mat.metallic;

    float D = brdf::distribution_ggx(NdotH, mat.roughness);
    float G = brdf::geometry_smith(NdotV, NdotL, mat.roughness);
    Vec3f F = brdf::fresnel_schlick(VdotH, F0);

    Vec3f spec = F * (D * G / (4.0f * NdotV * NdotL + 0.001f));
    Vec3f kD = (Vec3f(1.0f) - F) * (1.0f - mat.metallic);
    Vec3f diffuse = kD * mat.albedo / 3.14159265f;

    Vec3f brdf_val = diffuse + spec;

    // PDF for cosine-weighted hemisphere: NdotL / pi
    float pdf = NdotL / 3.14159265f;

    Ray bounce{rec.point, new_dir};
    Vec3f incoming = trace(bounce, depth + 1, rng, dist);

    return brdf_val * incoming * NdotL / (pdf + 1e-7f);
  }

  Vec3f sample_hemisphere(const Vec3f& N, std::mt19937& rng,
                          std::uniform_real_distribution<float>& dist) {
    float r1 = dist(rng);
    float r2 = dist(rng);
    float phi = 2.0f * 3.14159265f * r1;
    float cos_theta = std::sqrt(1.0f - r2);
    float sin_theta = std::sqrt(r2);

    Vec3f tangent;
    if (std::fabs(N.x) > 0.9f)
      tangent = cross(N, Vec3f(0, 1, 0)).normalized();
    else
      tangent = cross(N, Vec3f(1, 0, 0)).normalized();
    Vec3f bitangent = cross(N, tangent);

    return (tangent * std::cos(phi) * sin_theta + bitangent * std::sin(phi) * sin_theta +
            N * cos_theta)
        .normalized();
  }

  BVH bvh_;
  Material material_;
  Vec3f eye_ = {0, 0, 5};
  Vec3f target_ = {0, 0, 0};
  Vec3f up_ = {0, 1, 0};
  float fov_ = 60.0f;
  int width_ = 512, height_ = 512;
  int max_bounces_ = 4;
  int spp_ = 1;
  Vec3f background_ = {0.1f, 0.1f, 0.1f};
};

}  // namespace pt
