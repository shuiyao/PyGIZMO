/* Ion lookup table definitions */
#define MAXIONS 35
#define NHPTS 240 /* Number of n_h gridpoints in lookup table */
#define TPTS 140 /* Number of T gridpoints in lookup table */
#define GALPTS 11
/* Table limits */
#define NHLOW -9.0
#define DELTANH 0.05
#define TLOW 2.5
#define DELTAT 0.05
#define GALLOW 6.0
#define DELTAGAL 0.5

typedef struct ionStruct {
  char name[10];
  float lambda,fraction,Xsec,atomwt,bsys,alpha;
  int Zcolumn;
} ionStruct;

extern float nhl[NHPTS],tl[TPTS];
extern float fractab[MAXIONS][NHPTS][TPTS];
extern ionStruct Ion[MAXIONS];
