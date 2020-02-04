/*
    提取所有Scene中的instance信息
    包含：bbox_2d，category
*/
#include "CJsonObject/CJsonObject.hpp"
#include "lodepng/lodepng.h"
#include <bits/stdc++.h>

using namespace std;
using namespace neb;

string fileinfoStr;
CJsonObject fileinfo, cateinfo, bboxinfo;
char path[1000];
string pathStr;

string addZeros(string a){
    while (a.length() < 5)
        a = "0" + a;
    return a;
}
vector <unsigned char> inst, sem;

int instance(int h, int w){
    return int(inst[(h*1280 + w)*4]) * 256 + int(inst[(h*1280 + w)*4 + 1]);
}
int semantic(int h, int w){
    return sem[(h*1280 + w)*4];
}
string toString(int a){
    char s[15];
    sprintf(s, "%d", a);
    return string(s);
}
void saveJson(const char *filename, CJsonObject &obj){
    FILE *f = fopen(filename, "w");
    fprintf(f, "%s", obj.ToFormattedString().c_str());
    fclose(f);
}

const int MAX_ID = 3000;

int id_cate_map[MAX_ID][42], max_id;
int id_hmin[MAX_ID], id_hmax[MAX_ID], id_wmin[MAX_ID], id_wmax[MAX_ID], id_exist[MAX_ID];

int main(){
    FILE *f = fopen("/home/xuanrun/nas/Structured3D/meta/fileinfo.json", "r");
    char cstr[1024];
    while (fgets(cstr, 1024, f)) fileinfoStr += cstr;
    fclose(f);
    fileinfo = CJsonObject(fileinfoStr);
    string scene, room;

    int flag = 0, empty;
    while (fileinfo.GetKey(scene)){
        if (scene == "1910")
            flag = 1;
        if (!flag) continue;
        memset(id_cate_map, 0, sizeof id_cate_map);
        max_id = 0;
        while (fileinfo[scene].GetKey(room)){
            int size = fileinfo[scene][room].GetArraySize();
            for (int position = 0; position < size; ++position){
                memset(id_hmin, 0x7f, sizeof id_hmin);
                memset(id_hmax, 0x0, sizeof id_hmax);
                memset(id_wmin, 0x7f, sizeof id_wmin);
                memset(id_wmax, 0x0, sizeof id_wmax);
                memset(id_exist, 0x0, sizeof id_exist);
                sprintf(path, "/home/xuanrun/nas/Structured3D/dataset/scene_%s/2D_rendering/%s/perspective/full/%d/", addZeros(scene).c_str(), room.c_str(), position);
                printf("%s\n", path);
                pathStr = path;
                unsigned width, height;
                inst.clear();
                unsigned error = lodepng::decode(inst, width, height, (pathStr + "instance_8bit.png").c_str());
                assert(width == 1280 && height == 720);
                sem.clear();
                error = lodepng::decode(sem, width, height, (pathStr + "semantic_category.png").c_str());
                assert(width == 1280 && height == 720);
                for (int i = 0; i < height; ++i){
                    for (int j = 0, cate, id; j < width; ++j){
                        id = instance(i, j);
                        cate = semantic(i, j);
                        if (id != 65535){
                            max_id = max(max_id, id);
                            if (id >= MAX_ID){
                                cout << "ERROR!\n";
                                return 0;
                            }
                            id_cate_map[id][cate]++;
                            id_hmin[id] = min(id_hmin[id], i);
                            id_hmax[id] = max(id_hmax[id], i);
                            id_wmin[id] = min(id_wmin[id], j);
                            id_wmax[id] = max(id_wmax[id], j);
                            id_exist[id] = 1;
                        }
                    }
                }
                bboxinfo = CJsonObject("{}");
                for (int i = 0; i <= max_id; ++i)
                    if (id_exist[i]){
                        bboxinfo.AddEmptySubArray(toString(i));
                        bboxinfo[toString(i)].Add(toString(id_hmin[i]));
                        bboxinfo[toString(i)].Add(toString(id_wmin[i]));
                        bboxinfo[toString(i)].Add(toString(id_hmax[i]));
                        bboxinfo[toString(i)].Add(toString(id_wmax[i]));
                    }
                saveJson((pathStr + "bbox_2d.json").c_str(), bboxinfo);
            }
        }
        cateinfo = CJsonObject("{}");
        for (int i = 0; i <= max_id; ++i){
            int max_cnt = 0, max_index = -1;
            for (int j = 0; j < 41; ++j)
                if (id_cate_map[i][j] > max_cnt){
                    max_cnt = id_cate_map[i][j];
                    max_index = j;
                }
            if (max_cnt > 0)
                cateinfo.Add(toString(i), toString(max_index));
                // printf("%d %d\n", i, max_index);
        }
        if (max_id > 0){
            sprintf(path, "/home/xuanrun/nas/Structured3D/dataset/scene_%s/", addZeros(scene).c_str());
            pathStr = path;
            saveJson((pathStr + "idcate_map.json").c_str(), cateinfo);
        }
    }

    return 0;
}