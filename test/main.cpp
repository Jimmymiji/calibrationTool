#include  "angleSol.h"
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"


using namespace std;
using namespace cv;
double angle(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
void findSquares(const Mat& image, vector<vector<Point> >& squares)
{
    int thresh = 50, N = 5;
    squares.clear();

    Mat timg(image);
    medianBlur(image, timg, 5);
    Mat gray0(timg.size(), CV_8U), gray;
    vector<vector<Point> > contours;


    for (int c = 0; c < 3; c++)
    {
        int ch[] = { c, 0 };
        mixChannels(&timg, 1, &gray0, 1, ch, 1);


        for (int l = 0; l < N; l++)
        {

            if (l == 0)
            {
                Canny(gray0, gray, 5, thresh, 5);
                dilate(gray, gray, Mat(), Point(-1, -1));
            }
            else
            {
                gray = gray0 >= (l + 1) * 255 / N;
            }
            //imshow("canny",gray);
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
            imshow("gay",gray);
            vector<Point> approx;
            for (size_t i = 0; i < contours.size(); i++)
            {
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
                if (approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)))
                {
                    double maxCosine = 0;

                    for (int j = 2; j < 5; j++)
                    {
                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }
                    if (maxCosine < 0.3)
                        squares.push_back(approx);
                }
            }
        }
    }
}
bool descendingArea(const Rect& a,const Rect& b)  {return (a.area()>b.area());}
bool checkSquares(vector<vector<Point> >& squares, Rect& target)
{
    vector<Rect> rects;
    if(squares.size()==0)
    {
        return false;
    }
    for(int i = 0; i<squares.size();i++)
    {
        Rect temp = boundingRect(squares[i]);
        double ratio = (double)temp.height/temp.width;
        if(ratio > 0.4 && ratio < 1.1)
        rects.push_back(temp);
    }
    if(rects.size() == 0)
    {
        return  false;
    }
    sort(rects.begin(),rects.end(),descendingArea);
    target = rects[0];
    return true;
}
int main()
{
    int count = 0;
    VideoCapture cap("/dev/v4l/by-id/usb-HD_Camera_Manufacturer_Stereo_Vision_1_Stereo_Vision_1-video-index0");
    if(!cap.isOpened())
    {
        cout<<"camrea not opened"<<endl;
        return -1;
    }
    while(true)
    {
        Mat img;
        vector<vector<Point>> squares;
        cap >> img;
        imshow("original",img);
        waitKey(1);
        findSquares(img,squares);
        Rect target;
        if(checkSquares(squares,target))
        {
            AngleSolver ag;
            ag.setCameraMAtrix();
            ag.setRealWorldTargetS(210,150);
            vector<Point2f> input;
            input.push_back(Point2f(target.x, target.y));
            input.push_back(Point2f(target.x, target.y + target.height));
            input.push_back(Point2f(target.x + target.width, target.y));
            input.push_back(Point2f(target.x + target.width, target.y + target.height));
            ag.setImageTargetS(input,img);
            ag.getRotation_Translation_Matrix();
            ag.getPositionInfo();
            imshow("target",img);

        }
        else
        {
            cout<<"no target found"<<endl;
        }
        waitKey(30);
    }
}
