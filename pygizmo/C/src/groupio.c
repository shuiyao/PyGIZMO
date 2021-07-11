#include "tipsydefs.h"
#include "loadhdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NSKID_GRPS 100000

struct tipsy_header theader, soheader, galheader;
struct star_particle *gals, *sohalos;
struct skid_group *skid_grps;
struct fof_group *fofgrps;

int *sohid; /* Map from pidx -> HID */
int *galid; /* Map from pidx -> GID */
int *hosthid;

int Nfofgrps;

int read_tipsy_header(char *filename){
  FILE *fp;
  if(!(fp = fopen(filename, "r")))
    {printf("Can't open file %s! Force exit.\n", filename); exit(-1);}
  // TODO: Remove the MARK64
#ifdef MARK64
  printf("I am reading for 64 machine!\n");
  read_header(&theader, fp);
#else
  printf("I am reading for 32 machine!\n");
  if(fread((char *)&theader,sizeof(theader),1,fp) == 0)
    {printf("BREAK!\n"); exit(-1);}
#endif
  printf("*** Header Information ***\n");
  printf("header.time: %f\n", theader.time);
  printf("header.nbodies: %d\n", theader.nbodies);
  printf("header.ndim: %d\n", theader.ndim);
  printf("header.nsph: %d\n", theader.nsph);
  printf("header.ndark: %d\n", theader.ndark);
  printf("header.nstar: %d\n", theader.nstar);
  printf("**************************\n");
  fclose(fp);
  return 1;
}

int read_gtp(char *filename)
{
  FILE *fp;
  fp = fopen(filename, "r");
#ifdef MARK64
  printf("I am reading for 64 machine!\n");
  read_header(&galheader, fp);
#else
  printf("I am reading for 32 machine!\n");
  if(fread((char *)&galheader,sizeof(struct tipsy_header),1,fp) == 0) // PAD HERE
    {printf("EMPTY SOGTP FILE, BREAK!\n"); exit(-1);}
#endif
/*   headerinfo(&header); */
  printf("*** gtp Information ***\n");
  printf("galheader.nstar: %d\n", galheader.nstar);
  printf("**************************\n");

  if(galheader.nsph != 0) {fprintf(stdout, "BAD: NSPH != 0\n"); exit(-1);}
  if(galheader.ndark != 0) {fprintf(stdout, "BAD: NDARK != 0\n"); exit(-1);}
  if(galheader.nstar != 0) {
    gals = (struct star_particle *)
      malloc(galheader.nstar*sizeof(*gals));
    if(gals == NULL) {
      printf("<sorry, no memory for GALS particles, master>\n") ;
      return -1;
    }
    fread((char *)gals,sizeof(struct star_particle),
	  galheader.nstar,fp) ;
  }
  else{
    fprintf(stdout, "Warning: No galaxy in this snapshot!\n");
  }
  fclose(fp);
  printf("---> Reading .gtp done.\n");
  return 1;
}

int read_sogtp(char *filename)
{
  FILE *fp;
  fp = fopen(filename, "r");
#ifdef MARK64
  printf("I am reading for 64 machine!\n");
  read_header(&soheader, fp);
#else
  printf("I am reading for 32 machine!\n");
  if(fread((char *)&soheader,sizeof(struct tipsy_header)-4,1,fp) == 0) // PAD HERE
    {printf("EMPTY SOGTP FILE, BREAK!\n"); exit(-1);}
#endif
/*   headerinfo(&header); */
  printf("*** Sogtp Information ***\n");
  printf("soheader.nstar: %d\n", soheader.nstar);
  printf("**************************\n");

  if(soheader.nsph != 0) {fprintf(stdout, "BAD: NSPH != 0\n"); exit(-1);}
  if(soheader.ndark != 0) {fprintf(stdout, "BAD: NDARK != 0\n"); exit(-1);}
  if(soheader.nstar != 0) {
    sohalos = (struct star_particle *)
      malloc(soheader.nstar*sizeof(*sohalos));
    if(sohalos == NULL) {
      printf("<sorry, no memory for SOHALOS particles, master>\n") ;
      return -1;
    }
  }
  fread((char *)sohalos,sizeof(struct star_particle),
  	soheader.nstar,fp) ;
  fclose(fp);
  printf("---> Reading .sogtp done.\n");
  return 1;
}

int read_fofgroup(char *filename)
{
  FILE *fp;
  int i, j;
  char dummyline[500];
  int idummy;
  float mfof, mstar, mgal, sfr;
  double pos[3];
  int ngrps = 0;
  fofgrps = (struct fof_group *) malloc(MAX_NSKID_GRPS*sizeof(*fofgrps));
  i = 0;
  if(!(fp = fopen(filename, "r"))){
    fprintf(stderr, "Can't open FoF file! Force exit.\n"); exit(-1);}
  fscanf(fp, "%[^\n]", dummyline); // First line, header
  // #CpuID GroupID Mfof Mstar Mgal SFR x y z    
  while(1){
    fscanf(fp, "%d %d %f %f %f %f %lf %lf %lf", 
	   &idummy, &idummy, &mfof, &mstar, &mgal,
	   &sfr, &pos[0], &pos[1], &pos[2]);
    // We ignore dark halos (no stars / no gas)
    if(mgal >= 0){
      fofgrps[ngrps].gid = ngrps;
      fofgrps[ngrps].mgal = mgal;
      fofgrps[ngrps].mstar = mstar;
      fofgrps[ngrps].mfof = mfof;
      for (j=0;j<3;j++) fofgrps[ngrps].pos[j] = pos[j];
      ngrps ++;
      if(ngrps >= MAX_NSKID_GRPS){
	fprintf(stderr, "Number of FoF Groups exceeds limit.\n"); exit(-1);}
    }
    if(feof(fp)) break;
    i++;
  }
  fclose(fp);
  return ngrps;
}

int read_skid(int snapnum)
{
  FILE *inputfile;
  char skidname[200], skidbasename[200], grpname[200];
  int n, i, j, gid;
  char snapstr[4];
  char dummyline[500];
  float starmass, totmass, gasmass;
  int nskid_grps;

  get_snap_string(snapnum, snapstr);
  strcat(strcpy(skidname, skidbasename), snapstr);
  strcat(skidname, ".stat");
  strcat(strcpy(grpname, skidbasename), snapstr);
  strcat(grpname, ".grp");
  printf("Skidname: %s\n", skidname);
  printf("Grpname: %s\n", grpname);
  // Reading the .stat file
  skid_grps = (struct skid_group *) malloc(MAX_NSKID_GRPS*sizeof(*skid_grps));
  i = 0;
  if(!(inputfile = fopen(skidname, "r")))
    {printf("Can't open .stat file! Force exit.\n"); exit(-1);}
  while(1){
    fscanf(inputfile, "%d %d %f %f %f %[^\n]", 
	   &gid, &n, &totmass, &gasmass, &starmass, dummyline);
    skid_grps[i].gid = gid;
    skid_grps[i].mgas = gasmass;
    skid_grps[i].mstar = starmass;
    skid_grps[i].mgas = 0.;
    skid_grps[i].mstar = 0.;
    skid_grps[i].sfr = 0.;
    for (j=0;j<4;j++) skid_grps[i].metals[j] = 0.0;
    if(feof(inputfile)) break;
    i++;
  }
  nskid_grps = gid; // SHOULD BE THE LARGEST GID
  fclose(inputfile);
  printf("Reading .stat DONE. %d=%d groups read.\n", i, nskid_grps);
  return nskid_grps;
}

int read_grp(char *filename, int typecode)
{
  int i, nparts, dump;
  FILE *fp;
  if(!(fp = fopen(filename, "r")))
    {printf("Can't open .grp file! Force exit.\n"); exit(-1);}
  fscanf(fp, "%d\n", &nparts); // ONLY HERE: hid = Number of 
  if(typecode==0) nparts=theader.nsph;
  if(typecode==1) nparts=theader.nsph + theader.ndark;
  if(typecode==4) nparts=theader.nsph + theader.ndark + theader.nstar;
  galid = (int *)malloc(nparts*sizeof(int));
  i = 0;
  while(i<nparts){
    fscanf(fp, "%d\n", &dump);
    galid[i++] = dump;
    if(feof(fp)) break;
  }
  fclose(fp);
  for(i=0;i<nparts;i++)
    if(galid[i] < 0)
      galid[i] *= -1;
  fprintf(stdout, "---> Reading .grp done.\n");
  return 1;
}

int read_sogrp(char *filename, int typecode)
{
  int i, nparts, dump;
  FILE *fp;
  if(!(fp = fopen(filename, "r")))
    {printf("Can't open .sogrp file! Force exit.\n"); exit(-1);}
  fscanf(fp, "%d\n", &nparts); // ONLY HERE: hid = Number of 
  if(typecode==0) nparts=theader.nsph;
  if(typecode==1) nparts=theader.nsph + theader.ndark;
  if(typecode==4) nparts=theader.nsph + theader.ndark + theader.nstar;
  sohid = (int *)malloc(nparts*sizeof(int));
  i = 0;
  while(i<nparts){
    fscanf(fp, "%d\n", &dump);
    sohid[i++] = dump;
    if(feof(fp)) break;
  }
  fclose(fp);
  for(i=0;i<nparts;i++)
    if(sohid[i] < 0)
      sohid[i] *= -1;
  fprintf(stdout, "---> Reading .sogrp done.\n");
  return 1;
}

void read_sopar(char *filename)
{
  int i;
  FILE *fp;
  int gid, hid;
  
  hosthid = (int *) malloc(MAX_NSKID_GRPS * sizeof(int));
  i = 0;
  if(!(fp = fopen(filename, "r"))){
    fprintf(stderr, "Can't open .par file! Force exit.\n"); exit(-1);}
  while(1){
    fscanf(fp, "%d %d", &gid, &hid);
    hosthid[gid] = hid;
    if(feof(fp)) break;
    i++;
  }
  fclose(fp);
  fprintf(stdout, "---> Reading .par done.\n");
}
