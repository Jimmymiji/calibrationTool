#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>
#include <fstream>
#include "angleSol.h"
using namespace std;

void AngleSolver::rotateByZ(double x, double y, double thetaz, double& outx, double& outy)
{
	double x1 = x;//
	double y1 = y;
	double rz = thetaz * CV_PI / 180;
	outx = cos(rz) * x1 - sin(rz) * y1;
	outy = sin(rz) * x1 + cos(rz) * y1;
}

void AngleSolver::rotateByY(double x, double z, double thetay, double& outx, double& outz)
{
	double x1 = x;
	double z1 = z;
	double ry = thetay * CV_PI / 180;
	outx = cos(ry) * x1 + sin(ry) * z1;
	outz = cos(ry) * z1 - sin(ry) * x1;
}

void AngleSolver::rotateByX(double y, double z, double thetax, double& outy, double& outz)
{
	double y1 = y;
	double z1 = z;
	double rx = thetax * CV_PI / 180;
	outy = cos(rx) * y1 - sin(rx) * z1;
	outz = cos(rx) * z1 + sin(rx) * y1;
}
// rotate a point by a certain vector, return the point after rorating
Point3f AngleSolver::RotateByVector(double old_x, double old_y, double old_z, double vx, double vy, double vz, double theta)
{
	double r = theta * CV_PI / 180;
	double c = cos(r);
	double s = sin(r);
	double new_x = (vx*vx*(1 - c) + c) * old_x + (vx*vy*(1 - c) - vz*s) * old_y + (vx*vz*(1 - c) + vy*s) * old_z;
	double new_y = (vy*vx*(1 - c) + vz*s) * old_x + (vy*vy*(1 - c) + c) * old_y + (vy*vz*(1 - c) - vx*s) * old_z;
	double new_z = (vx*vz*(1 - c) - vy*s) * old_x + (vy*vz*(1 - c) + vx*s) * old_y + (vz*vz*(1 - c) + c) * old_z;
	return cv::Point3f(new_x, new_y, new_z);
}


void AngleSolver::setCameraMAtrix()
{
	FileStorage fs1;
	fs1.open("green.yml", FileStorage::READ);

	fs1["camera_matrix"] >> cameraMatrix;
	fs1["distortion_coefficients"]>> distortionCoefficients;
	cout<<"camera matrix: "<<endl<<cameraMatrix;
	cout<<"distortionMatrix: "<<endl<<distortionCoefficients;
	fs1.release();
}

void AngleSolver::setRealWorldTargetS(double width , double height)
{
    double Hwidth =  width/2;
	double Hheight = height/2;

	targetInWorld.push_back(Point3f(-Hwidth,Hheight,0));
	targetInWorld.push_back(Point3f(Hwidth,Hheight,0));
	targetInWorld.push_back(Point3f(-Hwidth,-Hheight, 0));
	targetInWorld.push_back(Point3f(Hwidth, -Hheight, 0));
}

bool AngleSolver::setImageTargetS(vector<cv::Point2f> input,Mat& img)
{
	//cout << "set Image target" << endl;
	if(input.size()!=4)
	{
		//cout<<"no valid input!"<<endl;
		return false;
	}
	Point2f vertices[4];
	int i = 0;
	for(vector<cv::Point2f>::iterator p = input.begin();p<input.end();p++)
	{
        vertices[i] = *p;
        i++;
	}
	Point2f lu, ld, ru, rd;

	sort(vertices, vertices+4, [](const Point2f & p1, const Point2f & p2) { return p1.x < p2.x; });
	if (vertices[0].y < vertices[1].y) {
		lu = vertices[0];
		ld = vertices[1];
	}
	else {
		lu = vertices[1];
		ld = vertices[0];
	}
	if (vertices[2].y < vertices[3].y) {
		ru = vertices[2];
		rd = vertices[3];
	}
	else {
		ru = vertices[3];
		rd = vertices[2];
	}

	targetInImage.clear();
	targetInImage.push_back(lu);
	targetInImage.push_back(ru);
	targetInImage.push_back(ld);
	targetInImage.push_back(rd);
	circle(img,lu,3,Scalar(255,0,0),3);
    circle(img,ru,3,Scalar(255,255,0),3);
	circle(img,ld,3,Scalar(255,0,255),3);
	circle(img,rd,3,Scalar(0,255,0),3);
	return true;

}

void AngleSolver::getRotation_Translation_Matrix()
{
	solvePnP(targetInWorld, targetInImage, cameraMatrix, distortionCoefficients, rotationMatrix, translationMatrix);
	//solvePnP(targetInWorld, targetInImage, cameraMatrix, distortionCoefficients, rotationMatrix, translationMatrix, false, CV_ITERATIVE);//
	//solvePnP(targetInWorld, targetInImage, cameraMatrix, distortionCoefficients, rotationMatrix, translationMatrix, false, CV_EPNP);//
}

void AngleSolver::getPositionInfo()
{
	double rm[9];
	cv::Mat rotM(3, 3, CV_64FC1, rm);
	Rodrigues(rotationMatrix, rotM);
	double r11 = rotM.ptr<double>(0)[0];
	double r12 = rotM.ptr<double>(0)[1];
	double r13 = rotM.ptr<double>(0)[2];
	double r21 = rotM.ptr<double>(1)[0];
	double r22 = rotM.ptr<double>(1)[1];
	double r23 = rotM.ptr<double>(1)[2];
	double r31 = rotM.ptr<double>(2)[0];
	double r32 = rotM.ptr<double>(2)[1];
	double r33 = rotM.ptr<double>(2)[2];
	this->thetaZ = atan2(r21, r11) / CV_PI * 180;
	this->thetaY = atan2(-1 * r31, sqrt(r32*r32 + r33*r33)) / CV_PI * 180;
	this->thetaX = atan2(r32, r33) / CV_PI * 180;
	// the coordinate of rune in camera
	this->tz = translationMatrix.ptr<double>(0)[2];
	this->ty = translationMatrix.ptr<double>(0)[1];
	this->tx = translationMatrix.ptr<double>(0)[0];
	double x = tx, y = ty, z = tz;
	rotateByZ(x, y, -1 * thetaZ, x, y);
	rotateByY(x, z, -1 * thetaY, x, z);
	rotateByX(y, z, -1 * thetaX, y, z);
	// the cooradinate of camera in world ,not useful
	Rx = -x;
	Ry = -y;
	Rz = -z;
	// what we need
	cout<<"X : "<<tx<<endl;
	cout<<"Y : "<<ty<<endl;
	cout<<"Z : "<<tz<<endl;
	cout<<"distance :"<<sqrt(tx*tx+ty*ty+tz*tz)<<endl;
	cout<<"=========================================================="<<endl;
}


void AngleSolver:: sendAns(Mat& img)//show the distance in consloe
{
	cout << "tx: " << this->tx << endl << "ty: " << this->ty << endl << "tz: " << this->tz << endl << "------" << endl;
    cout << "Rx: " << this->Rx << endl << "Ry: " << this->Ry << endl << "Rz: " << this->Rz << endl << "-----------------" << endl;
	int d = sqrt(tx*tx+ty*ty+tz*tz);
	cout << "x:"<<this->tx << "y:"<<this->ty<<"z:"<<this->tz<<endl;
	cout<<"distance: "<<d<<endl;
	putText(img,to_string(d),Point(100,100),FONT_HERSHEY_SIMPLEX, 1 , Scalar(0,0,255),3);
	imshow("fuck",img);
	waitKey(1);
}
