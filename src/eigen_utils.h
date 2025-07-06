/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@file
@brief Definition of the standard containers with Eigen allocator.
*/

#pragma once

#include <deque>
#include <map>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace Eigen {

template <typename T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T>
using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

template <typename K, typename V>
using aligned_map = std::map<K, V, std::less<K>,
                             Eigen::aligned_allocator<std::pair<K const, V>>>;

template <typename K, typename V>
using aligned_unordered_map =
    std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                       Eigen::aligned_allocator<std::pair<K const, V>>>;

inline Eigen::Affine3d getTransBetween(Eigen::Vector3d trans_start,
                                       Eigen::Quaterniond rot_start,
                                       Eigen::Vector3d trans_end,
                                       Eigen::Quaterniond rot_end) {
  Eigen::Translation3d t_s(trans_start(0), trans_start(1), trans_start(2));
  Eigen::Translation3d t_e(trans_end(0), trans_end(1), trans_end(2));

  Eigen::Affine3d start = t_s * rot_start.toRotationMatrix();
  Eigen::Affine3d end = t_e * rot_end.toRotationMatrix();

  Eigen::Affine3d result = start.inverse() * end;
  return result;
}

template <typename T>
inline Eigen::Matrix<T, 3, 3> SkewSymmetric(const Eigen::Matrix<T, 3, 1>& w) {
  Eigen::Matrix<T, 3, 3> w_x;
  w_x << T(0), -w(2), w(1), w(2), T(0), -w(0), -w(1), w(0), T(0);
  return w_x;
}

/** sorts vectors from large to small
 * vec: vector to be sorted
 * sorted_vec: sorted results
 * ind: the position of each element in the sort result in the original vector
 * https://www.programmersought.com/article/343692646/
 */
inline void sort_vec(const Eigen::Vector3d& vec, Eigen::Vector3d& sorted_vec,
                     Eigen::Vector3i& ind) {
  ind = Eigen::Vector3i::LinSpaced(vec.size(), 0, vec.size() - 1);  //[0 1 2]
  auto rule = [vec](int i, int j) -> bool {
    return vec(i) > vec(j);
  };  // regular expression, as a predicate of sort

  std::sort(ind.data(), ind.data() + ind.size(), rule);

  // The data member function returns a pointer to the first element of
  // VectorXd, similar to begin()
  for (int i = 0; i < vec.size(); i++) {
    sorted_vec(i) = vec(ind(i));
  }
}

inline Eigen::Vector3d R2ypr(const Eigen::Matrix3d& R) {
  Eigen::Vector3d n = R.col(0);
  Eigen::Vector3d o = R.col(1);
  Eigen::Vector3d a = R.col(2);

  Eigen::Vector3d ypr(3);
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * std::cos(y) + n(1) * std::sin(y));
  double r = atan2(a(0) * std::sin(y) - a(1) * std::cos(y),
                   -o(0) * std::sin(y) + o(1) * std::cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr / M_PI * 180.0;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(
    const Eigen::MatrixBase<Derived>& ypr) {
  typedef typename Derived::Scalar Scalar_t;

  Scalar_t y = ypr(0) / 180.0 * M_PI;
  Scalar_t p = ypr(1) / 180.0 * M_PI;
  Scalar_t r = ypr(2) / 180.0 * M_PI;

  Eigen::Matrix<Scalar_t, 3, 3> Rz;
  Rz << std::cos(y), -std::sin(y), 0, std::sin(y), std::cos(y), 0, 0, 0, 1;

  Eigen::Matrix<Scalar_t, 3, 3> Ry;
  Ry << std::cos(p), 0., std::sin(p), 0., 1., 0., -std::sin(p), 0., std::cos(p);

  Eigen::Matrix<Scalar_t, 3, 3> Rx;
  Rx << 1., 0., 0., 0., std::cos(r), -std::sin(r), 0., std::sin(r), std::cos(r);

  return Rz * Ry * Rx;
}

template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar> positify(
    const Eigen::QuaternionBase<Derived>& q) {
  // printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
  // Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(),
  // -q.z()); printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z()); return
  // q.template w() >= (typename Derived::Scalar)(0.0) ? q :
  // Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
  return q;
}

}  // namespace Eigen
