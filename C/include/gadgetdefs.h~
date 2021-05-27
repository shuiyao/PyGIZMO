#ifndef _GADGETDEFS_H
#define _GADGETDEFS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define _LARGEFILE_SOURCE
#define _FILE_OFFSET_BITS 64

#define  BOLTZMANN   1.3806e-16
#define  CM_PER_MPC  3.085678e24
#define  PROTONMASS  1.6726e-24
#define  ATOMIC_UNIT 1.660539e-24 
#define  HUBBLE      3.2407789e-18   /* in h/sec */
#define  GAMMA         (5.0/3)
#define  GAMMA_MINUS1  (2.0/3)
//#define  M_PI 3.14159265358979
//#define  XH   0.76

#define XH 0.76
#define XHE (1.0 - XH) / (4.0 * XH)

#define UNIT_L 3.085678e21
#define UNIT_V 1.e5
#define UNIT_M 1.989e43

//#define NMETALSHDF5 11
//#define NMETALS 4

#define Skip fread(&dummy,sizeof(dummy),1,fp)


struct gadget_dump
{
  int      npart[6];
  double   mass[6];
  double   time;
  double   redshift;
  int      flag_sfr;
  int      flag_feedback;
  unsigned int      npartTotal[6];
  int      flag_cooling;
  int      num_files;
  double   BoxSize;
  double   Omega0;
  double   OmegaLambda;
  double   HubbleParam; 

  int      flag_stellarage;
  int      flag_metals;

  unsigned int      npartTotalHighWord[6];

  int      flag_entropy_instead_u;
  int      flag_doubleprecision;

  int flag_potential;
  int flag_fH2;

  int flag_tmax;
  int flag_delaytime;

  int flag_lpt_ics;
  float flag_lpt_scalingcator;

  char     fill[32];  /* fills to 256 Bytes */
} ;

extern struct gadget_dump gheader ;

#endif
