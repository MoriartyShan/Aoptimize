//
// Created by moriarty on 10/17/20.
//

#ifndef BOARDDETECT_VALUEDIFFCOST_H
#define BOARDDETECT_VALUEDIFFCOST_H
#include "../Common/types.h"
#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
class ValueDiff {
private:
  const scalar _distance;
public:
  static ceres::CostFunction* Create(
      const scalar distance) {
    return (new ceres::AutoDiffCostFunction<ValueDiff, 1, 1, 1>(
        new ValueDiff(distance)));
  }

  ValueDiff(const scalar distance) :
      _distance(distance){};

  template <typename T>
  bool operator()(const T* const v1, const T* const v2, T* residuals) const {
    T diff = (*v1 - *v2) - (T)_distance;
//    *residuals = ceres::exp(diff * diff);
    *residuals = diff * (T)10000;
    return true;
  };
};

#endif //BOARDDETECT_VALUEDIFFCOST_H
