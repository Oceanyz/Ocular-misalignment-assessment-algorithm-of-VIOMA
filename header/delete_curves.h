#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#define IMG_SIZE 200 //400

static cv::Mat delete_curves( cv::Mat *edge, double mean_dist){

	int start_x=0;
	int end_x=edge->cols;
	int start_y=0;
	int end_y=edge->rows;
	
	std::vector<std::vector<cv::Point>> all_lines;
	all_lines.clear();
	std::vector<cv::Point> all_means;
	all_means.clear();

	std::vector<cv::Point> curve; //������ʱ��ż�⵽��curve
	int curve_idx=0;
	cv::Point mean_p;  //һ��curve��ƽ��λ��

	bool delete_curve = false; //ָʾ�Ƿ�ɾ���ö�curve

	bool check[IMG_SIZE][IMG_SIZE]; //�����ά���飬��ʼ��Ϊ�㣬�����ά����ĳߴ������ͼ�������ߴ�һ��

	for(int i=0; i<IMG_SIZE; i++)
		for(int j=0; j<IMG_SIZE; j++)
			check[i][j]=0;
	
	//delete short and straight lines 
	for(int i=start_x; i<end_x; i++)
	{
		for(int j=start_y; j<end_y; j++)
		{
			if(edge->data[(edge->cols*(j))+(i)]>0 && !check[i][j]) //����Ե��������δ��check��������Ϊ1�ĵ㣬��check�и�λ�õ�����Ϊ1�������ٿ���
			{
				int neig=0; //��ʼ���õ�ķ����ھ�Ϊ0��
				for(int k1=-1;k1<2;k1++)
				{
					for(int k2=-1;k2<2;k2++)
					{
						if(i+k2>=start_x && i+k2<end_x && j+k1>=start_y && j+k1<end_y) //�ж��ھӵ�û�г���ͼ��߽�
						{
							if(edge->data[(edge->cols*(j+k1))+(i+k2)]>0)
								neig++;
						}
					}
				}//����õ�9�����еķ����ھӵ�
				if(neig>=2)
				{
					check[i][j]=1;

					curve.clear();
					curve_idx=0;

					curve.push_back(cv::Point(i,j)); //���г��������ھӵĵ�ѹ��curve
					mean_p.x=i;
					mean_p.y=j;
					curve_idx++;

					int akt_idx=0;

					while(akt_idx<curve_idx)
					{
						cv::Point akt_pos=curve[akt_idx];
						for(int k1=-1;k1<2;k1++)
						{
							for(int k2=-1;k2<2;k2++)
							{
								if(akt_pos.x+k1>=start_x && akt_pos.x+k1<end_x && akt_pos.y+k2>=start_y && akt_pos.y+k2<end_y) //�ж��ھӵ�û�г���ͼ��߽�
								{
									if(!check[akt_pos.x+k1][akt_pos.y+k2] )
									{
										if( edge->data[(edge->cols*(akt_pos.y+k2))+(akt_pos.x+k1)]>0)
										{
											check[akt_pos.x+k1][akt_pos.y+k2]=1;
											mean_p.x += akt_pos.x+k1;
											mean_p.y += akt_pos.y+k2;
											curve.push_back(cv::Point(akt_pos.x+k1,akt_pos.y+k2));
											curve_idx++;
										}
									}
								}
							
							}//��for
						}//��for
						akt_idx++;

					}//while

					if(curve_idx<=5 && curve.size()<=5)  //���curve֮��ִ�������жϣ���curve�̣ܶ�ֱ��ɾ��
					{
						for(int r = 0;r<curve.size();r++)
						{
							int x = curve.at(r).x;
							int y = curve.at(r).y;

							edge->data[(edge->cols*(y))+(x)]=0; //��С��5�������㣨��Ϊ�ǹ����㣩���ڱ���ͼ������Ϊ0
							check[x][y]=0;
						}
					}//end if(curve_idx<=5 && curve.size()<=5)  
					
					if(curve_idx>5 && curve.size()>5) //��һ��ɾ��straight line
					{
						delete_curve = false;
						mean_p.x=floor(( double(mean_p.x)/double(curve_idx) )+0.5);
						mean_p.y=floor(( double(mean_p.y)/double(curve_idx) )+0.5);
						all_means.push_back(mean_p);//һ��curve��ƽ��λ��
						all_lines.push_back(curve);//��curveѹ��all_lines
						
						int num = 0; //���������������Լ���������ϵ�ĸ���
						float rate = 0; //rate���������������Լ���ĵ�ı���
						for(int r = 0;r<curve.size();r++) //ִ�о����ж�
						{
							if(  abs(mean_p.x-curve[r].x)<= mean_dist && abs(mean_p.y-curve[r].y) <= mean_dist) //�ж�curve��ֻҪ���ڵ㵽ƽ����ľ���С��ĳ����ֵ������Ϊ��ֱ�ߣ�Ȼ��ִ��ɾ������
							{
								num++;
							}
						}
						rate = 1.0f*num/curve.size();
						if (rate > 0.2) //����һ�����ʣ����������۽�ë�ȵĸ�����ɸ����߱�ɾ�� ���ʾ������Ϊ0.005
						{
							delete_curve=true;
						}
						if(delete_curve) //ִ��ɾ��straight line����
						{
							for(int r = 0;r<curve.size();r++)
							{
								int x = curve.at(r).x;
								int y = curve.at(r).y;

								edge->data[(edge->cols*(y))+(x)]=0; //�ڱ���ͼ������Ϊ0
								check[x][y]=0;
							}
						}
						
					}//end if(curve_idx>5 && curve.size()>5)
					
				}//if(neig>=2)
			}
		}//��forѭ������
	}//��forѭ������
	/*
	int size_lines = all_lines.size();
	if(size_lines<15)
	{
		for(int s = 0; s<all_lines.size();s++)
		{
			delete_curve = false;
			mean_p = all_means.at(s);
			curve = all_lines.at(s);
			int num = 0; //���������������Լ���������ϵ�ĸ���
			float rate = 0; //rate���������������Լ���ĵ�ı���
			for(int r = 0;r<curve.size();r++) //ִ�о����ж�
			{
				if(  abs(mean_p.x-curve[r].x)<= mean_dist && abs(mean_p.y-curve[r].y) <= mean_dist) //�ж�curve��ֻҪ���ڵ㵽ƽ����ľ���С��ĳ����ֵ������Ϊ��ֱ�ߣ�Ȼ��ִ��ɾ������,�������׵�����ɾ�������۽�ë���뵽ͫ����
				{
					num++;
				}
			}
			rate = 1.0f*num/curve.size();
			if (rate > 0.2) //����һ�����ʣ����������۽�ë�ȵĸ�����ɸ����߱�ɾ�� ���ʾ������Ϊ0.005,0.1,0.2
			{
				delete_curve=true;
			}
			if(delete_curve) //ִ��ɾ��straight line����
			{
				for(int r = 0;r<curve.size();r++)
				{
					int x = curve.at(r).x;
					int y = curve.at(r).y;

					edge->data[(edge->cols*(y))+(x)]=0; //�ڱ���ͼ������Ϊ0
					check[x][y]=0;
				}
			}
		}
	}//end if(size_lines<100)
	*/
	//����ϸ������
	for(int j=start_y; j<end_y; j++)
	{
		for(int i=start_x; i<end_x; i++)
		{
			int box[9];

			box[4]=(int)edge->data[(edge->cols*(j))+(i)];

			if(box[4])
			{
				if(j-1>=start_y && j+1<end_y && i-1>=start_x && i+1<end_x)
				{
					box[1]=(int)edge->data[(edge->cols*(j-1))+(i)];
					box[3]=(int)edge->data[(edge->cols*(j))+(i-1)];
					box[5]=(int)edge->data[(edge->cols*(j))+(i+1)];
					box[7]=(int)edge->data[(edge->cols*(j+1))+(i)];

					if((box[5] && box[7])) edge->data[(edge->cols*(j))+(i)]=0;
					if((box[5] && box[1])) edge->data[(edge->cols*(j))+(i)]=0;
					if((box[3] && box[7])) edge->data[(edge->cols*(j))+(i)]=0;
					if((box[3] && box[1])) edge->data[(edge->cols*(j))+(i)]=0;
					//ϸ������
				}
			}
		}
	}
	//too many neigbours���г���3���ھӵı��ص㱻ɾ������Ϊ�����������Ӷ��������
	for(int j=start_y; j<end_y; j++)
	{
		for(int i=start_x; i<end_x; i++)
		{
			int neig=0;
			for(int k1=-1;k1<2;k1++)
			{
				for(int k2=-1;k2<2;k2++)
				{
					if(i+k2>=start_x && i+k2<end_x && j+k1>=start_y && j+k1<end_y) //�ж��ھӵ�û�г���ͼ��߽�
					{
						if(edge->data[(edge->cols*(j+k1))+(i+k2)]>0)
							neig++;
					}
				}
			}

			if(neig>3)
				edge->data[(edge->cols*(j))+(i)]=0;

		}//��for
	}//��for

	return *edge;
}
