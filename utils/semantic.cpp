#include "lodepng/lodepng.h"
#include <cstdio>
#include <vector>
#include <cassert>
#include <string>
#include <iostream>

using namespace std;

unsigned char category_mapping[41][3] = {
       0, 0, 0,
       174, 199, 232,		
       152, 223, 138,		
       31, 119, 180, 		
       255, 187, 120,		
       188, 189, 34, 		
       140, 86, 75,  		
       255, 152, 150,		
       214, 39, 40,  		
       197, 176, 213,		
       148, 103, 189,		
       196, 156, 148,		
       23, 190, 207, 		
       178, 76, 76,  
       247, 182, 210,		
       66, 188, 102, 
       219, 219, 141,		
       140, 57, 197, 
       202, 185, 52, 
       51, 176, 203, 
       200, 54, 131, 
       92, 193, 61,  
       78, 71, 183,  
       172, 114, 82, 
       255, 127, 14, 		
       91, 163, 138, 
       153, 98, 156, 
       140, 153, 101,
       158, 218, 229,		
       100, 125, 154,
       178, 127, 135,
       120, 185, 128,
       146, 111, 194,
       44, 160, 44,  		
       112, 128, 144,		
       96, 207, 209, 
       227, 119, 194,		
       213, 92, 176, 
       94, 106, 211, 
       82, 84, 163,  		
       100, 85, 144
};
int color_mapping[200];
int cate_hash(unsigned char r, unsigned char g, unsigned char b){
    return ((int)r * 91 + (int)g * 23 + (int)b * 47) % 196;
}

int count[60];


int main(int argc, char **argv){
    vector<unsigned char> image;
    string filename = argv[1];
    string input = filename + "semantic.png";
    string output = filename + "semantic_category.png";
    assert(argc == 2);

    memset(color_mapping, -1, sizeof color_mapping);
    for (int i = 0; i < 41; ++i){
        int h = cate_hash(category_mapping[i][0], category_mapping[i][1], category_mapping[i][2]);
        color_mapping[h] = i;
    }

    unsigned width, height;
    // unsigned error = lodepng::decode(image, width, height, "/home/xuanrun/nas/Structured3D/dataset/scene_00000/2D_rendering/485142/perspective/full/0/semantic.png");
    unsigned error = lodepng::decode(image, width, height, input.c_str());
    if (image.size() == 0){
        FILE *f = fopen("fail.txt", "a");
        fprintf(f, "%s\n", input.c_str());
        fclose(f);
        printf("Failed to decode image.\n");
        return 0;
    }
    for (int i = 0; i < width * height; ++i){
        unsigned char r, g, b;
        r = image[i*4 + 0];
        g = image[i*4 + 1];
        b = image[i*4 + 2];
        int cate = color_mapping[cate_hash(r, g, b)];
        if (cate == -1){
            printf("Error in %s: invalid color.\n", filename.c_str());
            return 0;
        }
        count[cate]++;
        image[i*4 + 0] = cate;
        image[i*4 + 1] = cate;
        image[i*4 + 2] = cate;
        image[i*4 + 3] = 255;
    }
    error = lodepng::encode(output.c_str(), image, width, height);
    return 0;
}