#ifndef _GRID2D
#define _GRID2D

#ifndef GRID2D_PARAMS
#define NUMOFNODES_X 256
#define NUMOFNODES_Y 256
#define OMEGABARYON 0.045
#define XMIN -1.5
#define XMAX 7.0
#define YMIN 3.0
#define YMAX 8.0
#endif

struct NodesStruct {
  float x;
  float y;
  int z;
  double m;
  double met;
};

extern struct NodesStruct *nodes_array;

int load_grid(int nx, int ny);
int grid_count(float x, float y, float m, float metc, float meto);


#endif
