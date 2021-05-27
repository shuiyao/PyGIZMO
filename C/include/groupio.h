#ifndef _GROUPIO_H
#define _GROUPIO_H

extern struct star_particle *gals, *sohalos;
extern struct tipsy_header theader, soheader, galheader;
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

#endif
