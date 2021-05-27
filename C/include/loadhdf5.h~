#ifndef _LOADHDF5_H
#define _LOADHDF5_H

#define MAX_LEN_FILENAME 500
#define NMETALSHDF5 11

/* Variables */
extern char base_name[MAX_LEN_FILENAME];

extern double unit_Time;
extern double unit_Length;
extern double unit_Density;
extern double unit_Mass;
extern double unit_Velocity;
extern double unit_Temp;

extern int flag_read_dark;
extern int flag_read_star;
extern int *sohid; // From .sogrp file
extern int *galid; // From .grp file
extern int *hosthid; // From .par file

struct particle_data
{
  float  Pos[3];
  float  Vel[3];
  int ID;
  float Mass, Rho, Ne, Nh, Hsml, metal[NMETALSHDF5], fH2, Tmax;
  float Sfr; // StellarFormationTime for star particles
  float Temp;
  float DelayTime;
  int    Flag;
  int Key;
  float WindMass;
  float Mcloud;
  float LastSFTime;
#ifdef PHEW_TRACK_INFO
  float TrackInfo[4];
#endif  
};

extern struct particle_data *particles;
extern struct particle_data *P;
extern struct gadget_dump h;
extern struct star_particle *sohalos, *gals; // .sogtp and .gtp file
extern struct tipsy_header theader, soheader, galheader; // .sogtp and .gtp files
extern struct fof_group{
  int gid;
  float mstar;
  float mgal;
  float mfof;
  double pos[3];
}
  *fofgrps;
extern struct skid_group {
  int gid;
  float mgas;
  float mstar;
  double pos[3];
  double metals[4];
  double sfr;
}
  *skid_grps;

extern int Nfofgrps;

/* Functions */
int load_hdf5(char *snap);
int allocate_memory(void);
void get_snap_string(int snapnum, char *snapstr);
void cosmounits();
int read_tipsy_header(char *filename);
int read_gtp(char *filename);
int read_sogtp(char *filename);
int read_sogrp(char *filename, int typecode);
int read_grp(char *filename, int typecode);
void read_sopar(char *filename);
int read_fofgroup(char *filename);

#endif // _LOADHDF5_H
