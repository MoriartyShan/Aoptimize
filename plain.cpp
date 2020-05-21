#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <fstream>
#include "../Common/camera.h"
#include "../Common/utils.h"
#define RESUDUAL_NUM 12
#define DEFAULT_HEIGHT 1.4
DEFINE_int32(rand_seed, -1, "> 1 as seed, 1 is special,"
  " <= 0 for random");

std::string toString(double *matrix) {
  std::stringstream matrix_string;
  matrix_string.precision(8);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      matrix_string << matrix[i * 3 + j] << ",";
    }
    matrix_string << "\n";
  }
  return matrix_string.str();
}

//distance^2 of point P to line MN
//may be nagative value
template<typename T>
T GetSquaredDistance(const T *P, const T *M, const T* N) {
  T MN[2] = { N[0] - M[0], N[1] - M[1] };
  T MP[2] = { P[0] - M[0], P[1] - M[1] };
  T cross = MP[0] * MN[1] - MP[1] * MN[0];
  T mod_MN = MN[0] * MN[0] + MN[1] * MN[1];
  if (cross < (T)0) mod_MN = -mod_MN;
  return cross * cross / mod_MN;
}

//only for 3x3
template<typename T>
void MatrixMulti(const T *left, const T *right, T* result) {
  const int rown = 3, coln = 3;
  for (int i = 0; i < rown; i++) {
    for (int j = 0; j < coln; j++) {
      int idx = i * coln + j;
      result[idx] = (T)0;
      for (int m = 0; m < coln; m++) {
        result[idx] += left[i * coln + m] * right[m * coln + j];
      }
    }
  }
}

template<typename T>
std::array<T, 2> GetDeviation(const std::vector<std::array<T, 2>> &vectors, const std::array<T, 2>& ave) {
  std::array<T, 2> res = { (T)0 };
  for (auto &v : vectors) {
    T d = v[0] - ave[0];
    res[0] += (d * d);

    d = v[1] - ave[1];
    res[1] += (d * d);
  }
  res[0] /= (T)vectors.size();
  res[1] /= (T)vectors.size();
  return res;
}

template<typename T>
void InsertMatrixToPointer(const cv::Matx33d &m, T *elem) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      elem[i * 3 + j] = (T)m(i, j);
    }
  }
  return;
}

class ExtrinsicOptimize {
private:
  const static int LanePointNum = 2;
  using Lane = std::vector<cv::Point2d>;//start and end point of a lane
  std::vector<Lane> _lanes;
  cv::Matx33d _cameraK;

  template <typename T>
  void cross_dot(const std::array<std::array<T, 3>, LanePointNum>& l,
      const std::array<std::array<T, 3>, LanePointNum>& r, T *cross, T *dot) const {
    T left[2] = { l[0][0] - l[1][0], l[0][1] - l[1][1] };
    T right[2] = { r[0][0] - r[1][0], r[0][1] - r[1][1] };
    *cross = left[0] * right[1] - left[1] * right[0];
    *dot = left[0] * right[0] + left[1] * right[1];
  }

public:
  static ceres::CostFunction* Create(const cv::Matx33d& cameraK, const std::vector<Lane> &_points) {
    return (new ceres::AutoDiffCostFunction<ExtrinsicOptimize, RESUDUAL_NUM, 3>(
      new ExtrinsicOptimize(cameraK, _points)));
  }

  ExtrinsicOptimize(const cv::Matx33d& cameraK, const std::vector<Lane>& lanes) {
    _lanes = lanes;
    _cameraK = cameraK;
  }

  //extr[0:2] = rotation_vector, extr[3] is height
  template <typename T>
  bool operator()(const T* const extr, T* residuals) const {
    using PointT = std::array<T, 3>;
    using LaneT = std::array<PointT, LanePointNum>;
    const T height = (T)DEFAULT_HEIGHT;

    std::vector<LaneT> lanesT(_lanes.size());
    T Cam_R_Car[9], cameraK[9], m_matrix[9];
    std::array<T, 2> average = { (T)0 };;
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix), Cam_R_Car_M(Cam_R_Car);

    ceres::AngleAxisToRotationMatrix(extr, Cam_R_Car_M);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);


    for (int i = 0; i < _lanes.size(); i++) {
      for (int j = 0; j < LanePointNum; j++) {
        T uv[2] = { (T)_lanes[i][j].x, (T)_lanes[i][j].y };
        Camera::GetPoint3d(m_matrix, uv, height, lanesT[i][j].data());
      }
    }
    T base_x = ceres::abs(lanesT[1][0][0] - lanesT[0][0][0]);
    T base_z = ceres::abs(lanesT[0][1][2] - lanesT[0][0][2]);
    T base[2] = { lanesT[0][0][0],  lanesT[0][0][2] };
    for (int i = 0; i < _lanes.size(); i++) {
      for (int j = 0; j < LanePointNum; j++) {
        lanesT[i][j][0] = (lanesT[i][j][0] - base[0]) / base_x;
        lanesT[i][j][2] = (lanesT[i][j][2] - base[1]) / base_z;
        //LOG(ERROR) << "[" << i << "][" << j << "] = [" << lanesT[i][j][0] << "," << lanesT[i][j][2] << "]";
      }
    }
    std::vector<std::pair<int, int>> lines;
    lines.reserve(LanePointNum * _lanes.size());
    for (int i = 0; i < _lanes.size(); i++) {
      for (int j = 0; j < LanePointNum; j++) {
        lines.emplace_back(i, j);
      }
    }

    std::sort(lines.begin(), lines.end(),
        [&lanesT](const std::pair<int, int>& l, const std::pair<int, int> &r) {
      if (lanesT[l.first][l.second][2] < lanesT[r.first][r.second][2]) return true;
      else return false;
    });

    auto begin = lines.begin();
    auto end = begin;
    std::advance(end, 4);
    std::sort(begin, end,
        [&lanesT](const std::pair<int, int>& l, const std::pair<int, int> &r) {
      if (lanesT[l.first][l.second][0] < lanesT[r.first][r.second][0]) return true;
      else return false;
    });
    std::sort(end, lines.end(),
        [&lanesT](const std::pair<int, int>& l, const std::pair<int, int> &r) {
      if (lanesT[l.first][l.second][0] < lanesT[r.first][r.second][0]) return true;
      else return false;
    });
    for (int i = 0; i < lines.size(); i++) {
       LOG(ERROR) << "[" << lines[i].first << "][" << lines[i].second << "] = ["
             << lanesT[lines[i].first][lines[i].second][0] << ","
             << lanesT[lines[i].first][lines[i].second][2] << "]";
    }
    for (int i = 0; i < lines.size(); i++) {



    }





#if 0
    std::vector<T> dx(lanesT.size()), dz(lanesT.size());
    std::vector<T> interv(lanesT.size() - 1);
    for (int i = 0; i < lanesT.size(); i++) {
      dx[i] = lanesT[i][0][0] - lanesT[i][LanePointNum - 1][0];
      dz[i] = lanesT[i][0][2] - lanesT[i][LanePointNum - 1][2];
      if (i < lanesT.size() - 1) {
        T dist[4] = {
          lanesT[i + 1][0][0] - lanesT[i][0][0],
          lanesT[i + 1][0][0] - lanesT[i][1][0],
          lanesT[i + 1][1][0] - lanesT[i][0][0],
          lanesT[i + 1][1][0] - lanesT[i][1][0],
        };
        T min = dist[0], max = min;
        for (int j = 1; j < 4; j++) {
          if (min > dist[j]) {
            min = dist[j];
          } else if (max < dist[j]) {
            max = dist[j];
          }
        }
        //interv[i] = max - min;
        residuals[i + 8] = max - min;
        interv[i] = (ceres::abs(max) + ceres::abs(min)) / (T)2;
      }
      residuals[i] = ceres::abs(dx[i]);
      residuals[i + 4] = (T)100 * ceres::abs(dx[i] / dz[i]);
    }
    residuals[11] = (T)0;
    for (int i = 0; i < interv.size(); i++) {
      for (int j = i + 1; j < interv.size(); j++) {
        residuals[11] += ceres::abs(interv[i] - interv[j]);
      }
    }
#endif
    return true;
  }

};


cv::Matx33d eulerAnglesToRotationMatrix(const cv::Vec3d &theta) {
  // Calculate rotation about x axis
  cv::Matx33d R_x(
    1, 0, 0,
    0, std::cos(theta[0]), -std::sin(theta[0]),
    0, std::sin(theta[0]), std::cos(theta[0])
  );
  // Calculate rotation about y axis
  cv::Matx33d R_y(
    std::cos(theta[1]), 0, std::sin(theta[1]),
    0, 1, 0,
    -std::sin(theta[1]), 0, std::cos(theta[1])
  );
  // Calculate rotation about z axis
  cv::Matx33d R_z(
    std::cos(theta[2]), -std::sin(theta[2]), 0,
    std::sin(theta[2]), std::cos(theta[2]), 0,
    0, 0, 1
  );
  // Combined rotation matrix
  cv::Matx33d R = R_z * R_y * R_x;
  return R;
}


int main() {
  std::string camera_file("D:\\Projects\\BoardDetect\\resources\\hardwares\\C.yaml");

  unsigned int rand_seed = FLAGS_rand_seed;
  if (FLAGS_rand_seed <= 0) {
    rand_seed = time(0);
  }
  srand(rand_seed);

  auto camera_ptr = Camera::create(camera_file);
  std::vector<std::vector<cv::Point2d>> points(4), undistort_points(4);

  //calculate points
  const int total_number = 2;
  points[0].resize(total_number);
  points[1].resize(total_number);
  points[2].resize(total_number);
  points[3].resize(total_number);


#if 0
  points[0][0] = cv::Point2d(65, 749);
  points[0][total_number - 1] = cv::Point2d(909, 506);

  points[1][0] = cv::Point2d(675, 765);
  points[1][total_number - 1] = cv::Point2d(963, 508);

  points[2][0] = cv::Point2d(1410, 755);
  points[2][total_number - 1] = cv::Point2d(1021, 505);

  points[3][0] = cv::Point2d(1811, 666);
  points[3][total_number - 1] = cv::Point2d(1067, 507);
#else
  points[0][0] = cv::Point2d(65, 749);
  points[0][total_number - 1] = cv::Point2d(935, 501);

  points[1][0] = cv::Point2d(675, 765);
  points[1][total_number - 1] = cv::Point2d(966, 505);

  points[2][0] = cv::Point2d(1410, 755);
  points[2][total_number - 1] = cv::Point2d(1026, 509);

  points[3][0] = cv::Point2d(1811, 666);
  points[3][total_number - 1] = cv::Point2d(1097, 509);
#endif

  auto FillPoints = [&points](const int id) {
    auto &line = points[id];
    const int size = line.size() - 1;
    cv::Point2d step = (line[size] - line[0]) / (size);
    LOG(ERROR) << "step = " << step;
    for (int i = 1; i < size; i++) {
      line[i] = line[i - 1] + step;
    }
  };
  //cv::Mat img = cv::imread("D:\\Projects\\Documents\\20200518_D\\undistort/undist_2005181158164399.png", cv::IMREAD_UNCHANGED);
  for (int i = 0; i < 4; i++) {
    FillPoints(i);
    //cv::line(img, points[i][0], points[i][total_number - 1], CV_RGB(255, 0, 0));
  }
  //cv::imwrite("D:\\Projects\\Documents\\20200518_D\\undistort/undist_undist_2005181158164399.png", img);

  for (int i = 0; i < 4; i++) {
    std::sort(points[i].begin(), points[i].end(), [](const cv::Point2d &l, const cv::Point2d &r) {
      if (l.y > r.y || (l.y == r.y && l.x < r.x)) return true;
      else return false;
    });
    MLOG() << points[i][0] << "-" << points[i][1];
    undistort_points[i] = points[i];
#if 0
    cv::undistortPoints(points[i], undistort_points[i], camera_ptr->Intrinsic(), \
      camera_ptr->Distortion(), cv::noArray(), camera_ptr->Intrinsic());
#endif
  }
  std::vector<double> res = { 0,0,0 };
  cv::Vec3d angles(-0.04, 0.1, 0);
  //(0.9959613358657007, -0.02060452005581651,
  //-0.08738690527692515, 0.02373564188500315,
  //  0.9991073673713584, 0.03494406628571539,
  //  0.08658889515934312, -0.0368771232275925,
  //  0.9955613697896987);//
  cv::Matx33d rotation = eulerAnglesToRotationMatrix(angles);

  MLOG() << rotation;
  rotation = camera_ptr->Intrinsic() * rotation.inv();
  for (int i = 0; i < 4; i++) {
    std::stringstream ss;
    ss << "line [" << i << "]\n";
    for (int j = 0; j < undistort_points[i].size(); j++) {
      double uv[2] = { undistort_points[i][j].x, undistort_points[i][j].y };
      double p3d[3];
      Camera::GetPoint3d(rotation.val, uv, DEFAULT_HEIGHT, p3d);
      ss << "[" << p3d[0] << "," << p3d[1] << "," << p3d[2] << "]\n";
    }
    LOG(ERROR) << ss.str();
  }
  KEEP_CMD_WINDOW();

#if 0
  auto cost_func = ExtrinsicOptimize::Create(camera_ptr->Intrinsic(), undistort_points);
  ceres::Problem problem;

  problem.AddResidualBlock(cost_func, nullptr, res.data());
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 100000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  LOG(ERROR) << "FindRotation:" << summary.BriefReport();
#endif

  cv::Vec3d r(res[0], res[1], res[2]);
  cv::Matx33d R;
  cv::Rodrigues(r, R);
  LOG(ERROR) << "best result = [" << res[0] << "," << res[1] << "," << res[2] << "] ";
  LOG(ERROR) << "rotation = \n" << R;
  LOG(ERROR) << "car_R_cam = \n" << R.inv();

  cv::Matx33d m_Matrix_ceres = camera_ptr->Intrinsic() * R;
  LOG(ERROR) << "Cam_R_Car = \n" << toString(R.val);
  LOG(ERROR) << "m matrix = \n" << toString(m_Matrix_ceres.val);

  for (int i = 0; i < 4; i++) {
    std::stringstream ss;
    ss << "line [" << i << "]\n";
    for (int j = 0; j < undistort_points[i].size(); j++) {
      double uv[2] = { undistort_points[i][j].x, undistort_points[i][j].y };
      double p3d[3];
      Camera::GetPoint3d(m_Matrix_ceres.val, uv, DEFAULT_HEIGHT, p3d);
      ss << "[" << j << "]" << p3d[0] << "," << p3d[1] << "," << p3d[2] << "]\n";
    }
    LOG(ERROR) << ss.str();
  }

  ExtrinsicOptimize opt(camera_ptr->Intrinsic(), undistort_points);
  double residuals[RESUDUAL_NUM];

  opt(res.data(), residuals);
  std::string pit;
  for (int i = 0; i < RESUDUAL_NUM; i++) {
    pit += (std::to_string(residuals[i]) + ",");
  }
  LOG(ERROR) << pit;


  KEEP_CMD_WINDOW();

  return 0;
}