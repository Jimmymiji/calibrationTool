#ifndef AS
#define AS
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
using namespace std;
using namespace cv;
class AngleSolver
{
public:
	vector<cv::Point2f> targetInImage;//the coners of target rect in image, pixels
	vector<cv::Point3f> targetInWorld;//the target in the reall world,we denote the center of the target as the origin of world coordinate
	Mat distortionCoefficients;//the distortion matrix of camera;
	Mat cameraMatrix;//the camera matrix of camera
	double Rx, Ry, Rz; // camera pisition in real world coordinate
	double tx, ty, tz;// rune position in camera coordinate
	double thetaX, thetaY, thetaZ;// the eular angles
	double sx, sy, sz;// position between shooter and camera
	double speed;//bullet speed
	Mat rotationMatrix;
	Mat translationMatrix;
	double shooterAngleYaw, shooterAnglePitch; // need a function to initialize !!!!

	AngleSolver()
	{
		rotationMatrix = cv::Mat::zeros(3, 1, CV_64FC1);
		translationMatrix = cv::Mat::zeros(3, 1, CV_64FC1);
	}
	void rotateByZ(double x, double y, double thetaz, double& outx, double& outy);
	void rotateByY(double x, double z, double thetay, double& outx, double& outz);
	void rotateByX(double y, double z, double thetax, double& outy, double& outz);
	Point3f RotateByVector(double old_x, double old_y, double old_z, double vx, double vy, double vz, double theta);
	void setCameraMAtrix();
	void setRealWorldTargetS(double width,double height);
	bool setImageTargetS(vector<cv::Point2f> input, Mat& img);
	void getRotation_Translation_Matrix();
	void getPositionInfo();
	void sendAns(Mat& img);
};

#endif