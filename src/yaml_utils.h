/*
 * Coco-LIC: Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry using Non-Uniform B-spline
 * Copyright (C) 2023 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <yaml-cpp/yaml.h>
#include <iostream>

namespace yaml {

template <typename T>
inline T GetValue(const YAML::Node& node, const std::string& descri,
                    T intial = T(0)) {
  T value = intial;
  if (node[descri]) {
    value = node[descri].as<T>();
  } else {
    // std::cout << descri << " set as 0\n";
  }
  return value;
}

template <typename T>
inline bool GetValues(const YAML::Node& node, const std::string& descri,
                      size_t size, std::vector<T>& params_vec) {
  params_vec.resize(size);

  if (node[descri]) {
    for (size_t i = 0; i < size; i++) {
      params_vec.at(i) = node[descri][i].as<T>();
    }
    return true;
  } else {
    std::cout << "Note [" << descri << "] is not provided.\n";
    return false;
  }
}

inline std::string GetString(const YAML::Node& node, const std::string& descri,
                             std::string intial = "") {
  std::string str = intial;
  if (node[descri]) {
    str = node[descri].as<std::string>();
  } else {
    std::cout << "Note [" << descri << "] set as an empty string\n";
  }
  return str;
}

inline bool GetBool(const YAML::Node& node, const std::string& descri,
                    bool ret = false) {
  if (node[descri]) {
    ret = node[descri].as<bool>();
  }
  return ret;
}

}  // namespace yaml
