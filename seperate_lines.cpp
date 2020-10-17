//
// Created by moriarty on 10/17/20.
//

#include "PointAtAxisLineCost.h"
#include "PointsAxisDistance.h"
#include "../Common/camera.h"
#include "ValueDiffCost.h"

void read_points(
    const Camera::CameraPtr &camera_ptr,
    const std::string &points_file,
    std::vector<std::vector<cv::Point2d>> &lines) {
  cv::FileStorage config(points_file, cv::FileStorage::READ);
  CHECK(config.isOpened()) << "points file:" << points_file
                           << " load fail, make sure it is exist";
  bool distorted = false;
  std::string type = config["points_type"].string();
  if (type == "Raw") {
    distorted = true;
  } else if (type == "Undistorted") {
    distorted = false;
  } else {
    LOG(FATAL) << "points_type = " << type << ", invalid";
  }

  lines.clear();
  for (int i = 1; ; i++) {
    std::string name = "image" + std::to_string(i);
    cv::FileNode &&node = config[name];
    if (node.isNone()) {
      break;
    } else {
      int cur_size = lines.size();
      MLOG() << "read " << name;
      lines.reserve(cur_size + 4);

      for (int i = 0; ; i++) {
        std::string node_name = "line" + std::to_string(i + 1);
        const cv::FileNode &sub_node = node[node_name];
        if (sub_node.isNone()) {
          LOG(ERROR) << "nothing named " << node_name << ", only " << i
                     << " line input";
          break;
        }
        LOG(ERROR) << node_name;
        lines.emplace_back();
        cv::read(sub_node, lines[cur_size + i]);
        LOG(ERROR) << "read sub node: " << node_name;
        std::string pppp;
        for (int j = 0; j < lines[cur_size + i].size(); j++) {
          pppp += ("[" + std::to_string(lines[cur_size + i][j].x) + ","
                   + std::to_string(lines[cur_size + i][j].y) + "]");
        }
        LOG(ERROR) << pppp;
      }
    }
  }
//  exit(0);
//  if (distorted) {
//    const double boundary = 30;
//    const double u_range[2] = { boundary, params.camera_ptr->ImageWidth() - boundary };
//    const double v_range[2] = { boundary, params.camera_ptr->ImageHeight() - boundary };
//    std::vector<cv::Point2d> tmp;
//    LOG(ERROR) << "Undisting points";
//    for (int i = 0; i < points.size(); i++) {
//      params.camera_ptr->UndistortPoints(points[i], tmp);
//      for (int j = 0; j < tmp.size(); j++) {
//        if (tmp[j].x > u_range[0] && tmp[j].x < u_range[1]
//            && tmp[j].y > v_range[0] && tmp[j].y < v_range[1]) {
//          //inside this contour after undistort, a good one
//        } else {
//          LOG(FATAL) << "Should change point " << points[i][j]
//                     << " on line[" << i << "] more inside";
//        }
//      }
//      points[i] = tmp;
//    }
//  }

//  auto nnode = config["start"];
//  if (!nnode.isNone()) {
//    cv::read(nnode, params.res);
//  }
  config.release();

}
double my_log(const double a, const double b) {
  return std::log(a) / std::log(b);
}

#define USE_LOG(a) my_log( a, 10)
void load_points(const Camera::CameraPtr &camera_ptr,
                 const std::string &points_file,
                 std::vector<std::vector<cv::Point2d>> &lines) {
  const int points_number = 80;
  read_points(camera_ptr, points_file, lines);
  std::vector<double> places;

#define USE_LOG(a) my_log( a, 10)
  places.reserve(points_number);

  for (int i = 0; i < points_number; i++) {
    places.emplace_back(USE_LOG(i + 1));
  }

  auto FillPoints = [&lines, &points_number, &places](const int id) {
    auto &line = lines[id];
    if (line.empty()) {
      return;
    }
    std::sort(line.begin(), line.end(), [](const cv::Point2d &l, const cv::Point2d &r) {
      if (l.y > r.y || (l.y == r.y && l.x < r.x)) return true;
      else return false;
    });

    line.resize(points_number);
    cv::Point2d max_dist = (line[1] - line[0]);
    for (int i = 1; i < points_number; i++) {
      line[i] = line[0] + max_dist * places[i] / places.back();
    }

    std::sort(line.begin(), line.end(), [](const cv::Point2d &l, const cv::Point2d &r) {
      if (l.y > r.y || (l.y == r.y && l.x < r.x)) return true;
      else return false;
    });
    return;
  };
  cv::Mat img = cv::imread("/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1022/undist.png", cv::IMREAD_UNCHANGED);
  for (int i = 0; i < lines.size(); i++) {
    FillPoints(i);
    cv::line(img, lines[i][0], lines[i][points_number - 1], CV_RGB(255, 0, 0));
  }
  cv::imwrite("/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1022/undist_undist_2005181158164399.png", img);
}
int main() {
  std::string camera_path = "/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1022/camera.yaml";
  std::string points_path = "/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1022/points.yaml";
  auto camera_ptr = Camera::create(camera_path);
  const cv::Matx33d &cameraK = camera_ptr->Intrinsic();

  std::vector<std::vector<cv::Point2d>> lines;
  load_points(camera_ptr, points_path,lines);

  ceres::Problem problem;

  cv::Vec3d cam_r_car;
  double height = 4;
  double positions[5] = {-16.01, -12.21, -8.15, -4.1, -0.15};
  for (int i = 0; i < lines.size(); i++) {
    for (int j = 0; j < lines[i].size(); j++) {
      auto cost_func = PointAtAxisLine::Create(lines[i][j], cameraK, 0);
      problem.AddResidualBlock(cost_func, nullptr, cam_r_car.val, &height, positions + i);
//      if (i != 0) {
//        const double distance = positions[i] - positions[i - 1];
//        for (int k = 0; k < lines[i - 1].size(); k++) {
//          auto cost_func_2 = PointAxisDistance::Create(lines[i][j], lines[i- 1][k], cameraK, distance, 0);
//          problem.AddResidualBlock(cost_func_2, nullptr, cam_r_car.val, &height);
//        }
//      }
    }
  }

  for (int i = 1; i < 5; i++) {
    auto cost_func = ValueDiff::Create(positions[i] - positions[i - 1]);
    problem.AddResidualBlock(cost_func, nullptr, positions + i, positions + i - 1);
  }


  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
//  options.logging_type = log_type;
  //options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  MLOG() << summary.BriefReport();

  cv::Matx33d cam_R_car;
  cv::Rodrigues(cam_r_car, cam_R_car);
  cv::Matx33d car_R_cam = cam_R_car.inv();

  LOG(ERROR) << cam_r_car << height;
  LOG(ERROR) << std::endl << cam_R_car;

  camera_ptr->set_Car_R_Cam(car_R_cam);
  camera_ptr->set_camera_height(height);
  for (int i = 0; i < lines.size(); i++) {
    LOG(ERROR) << "line:" << positions[i];
    for (int j = 0; j < lines[i].size(); j++) {
      auto p = camera_ptr->GetPoint3d(lines[i][j]);
//      LOG(ERROR) << lines[i][j] << p << std::endl;
    }
  }

  return 0;

}