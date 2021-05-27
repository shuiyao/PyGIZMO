/* Calculate the Mass fraction of a certain Ion */
/* fion ionid redshift n_h[cm^-3] T[K] */
/* IonID   0  1  2  3  4  5  6  7  8 */
/* IonName HI  HeII  CIII  CIV  OIV  OVI  NeVIII  MgII  SiIV */

#ifdef IONS

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <string.h>
#include "tipsydefs.h"
#include "proto.h"
#include "iontab.h"
#include <unistd.h>

#define MHYDR	1.673e-24
#define KBOLTZ  1.381e-16

/* ionStruct Ion[MAXIONS]; */
double redshift;
char ionbkgd[15];
int nions;
float nhl[NHPTS],tl[TPTS];
float fractab[MAXIONS][NHPTS][TPTS];

/* Read from file specions.dat:
   ion_name  rest_freq  osc_strength  atom_wt  solar_abundance  rel_abundance
 * solar abundance mass fractions are taken from Arnett, "Supernovae 
	and Nucleosynthesis" (1996) Table A.1 column 4,
	which in turn is from Anders & Grevesse 1989 and others.
 * rel_abundance is in dex vs.solar (not used in specgen)
*/
int InitIons(double zsnap)
{
  int i;
  char line[80],prefix[150],specionfilename[180];
  FILE *specfile;

  redshift = zsnap;
  // TODO: Change path
  sprintf(prefix,"/home/shuiyao/temp/specexbin/ionfiles/");
  /* Default: specions_i9.dat */
  sprintf(specionfilename,"%sspecions_i9.dat",prefix);
  if( (specfile = fopen(specionfilename,"r")) == NULL ) specfile = fopen(specionfilename,"r");
  if( specfile == NULL ) {
    fprintf(stderr,"cannot find specion file anywhere\n");
    exit(-1);
  }

  i = 0;
  while( fgets(line,80,specfile) != NULL ) {
    if( strstr(line,"#") != NULL ) continue;
    if( i >= MAXIONS ) break;
    sscanf(line,"%10s %g %g %g %g %d %g",Ion[i].name,&Ion[i].lambda,&Ion[i].Xsec,&Ion[i].atomwt,&Ion[i].fraction,&Ion[i].Zcolumn,&Ion[i].alpha);
    i++;
  }
  nions = i;
  fclose(specfile);

#ifdef VERBOSE  
  fprintf(stderr,"Processing %d ions from specions.dat:\n",nions);
#endif  
  for( i=0; i<nions; i++ ) {
    Ion[i].bsys = sqrt(2.*KBOLTZ/(MHYDR*Ion[i].atomwt))/1.e5;
    Ion[i].Xsec *= 2.648e-2*Ion[i].lambda*1.e-13;
#ifdef VERBOSE      
    fprintf(stderr,"%5d %10s %12.6g %12.6g %10.5g %10.5g %10.5g % 3.1f % 2d\n",i,Ion[i].name,Ion[i].lambda,Ion[i].Xsec,Ion[i].atomwt,Ion[i].fraction,Ion[i].bsys*sqrt(1.e4),Ion[i].alpha,Ion[i].Zcolumn);
#endif    
  }

  strcpy(ionbkgd,"HMQG");
#ifdef UVBKG_HM12
  strcpy(ionbkgd,"HM12");
#endif
  load_fraction_tables();
  return 0;
}

/* ------------------------------------------------------
   Calculate log(fractions) of HI,HeI,HeII,CII,CIV,O6,N5,SiIV
   by bilinear table interpolation in hydrogen density 
   and temperature. UH August 1996
   ------------------------------------------------------*/
double IonFrac(float temp, float density, int ionid)
{
  int inh,itemp;
  float y1,y2,y3,y4,t,u;
  float n_h;
  double ionfraction;
  static float nhmax=0.000001,nhmin=999,tmax=11.,tmin=1.e+07;
  n_h = density/MHYDR;
  if(n_h<nhmin) nhmin=n_h;
  if(n_h>nhmax) nhmax=n_h;
  if(temp<tmin) tmin=temp;
  if(temp>tmax) tmax=temp;
  /* --------------------------------------------------------
   * n_h is cgs number density of Hydrogen
   * temp is temperature in K
   * --------------------------------------------------------*/
 
  inh = (log10(n_h) - NHLOW)/DELTANH ;
  itemp = (log10(temp) - TLOW)/DELTAT ;

  /*  printf("ionfrac.c: n_h = %g, temp = %f\n", n_h, temp); */
  /*  printf("ionfrac.c: MHYDR = %g\n", MHYDR); */
  /*  printf("ionfrac.c: inh = %d, itemp = %d\n", inh, itemp); */

  /* if((inh>NHPTS-2 || itemp>TPTS-2|| inh<0 || itemp <0) && ionid == 0 ){
     fprintf(stderr,"%d: n_h=%g %d, T=%g %d\n",outbound,n_h,inh,temp,itemp);
     outbound++;
     }*/
  if(inh>NHPTS-2) inh=NHPTS-2;
  if(itemp>TPTS-2)itemp=TPTS-2;
  if(inh<0)inh=0;
  if(itemp<0)itemp=0;
  t = (log10(n_h)-nhl[inh])/DELTANH;
  u = (log10(temp)-tl[itemp])/DELTAT;

  y1 = fractab[ionid][inh][itemp];
  y2 = fractab[ionid][inh+1][itemp];
  y3 = fractab[ionid][inh+1][itemp+1];
  y4 = fractab[ionid][inh][itemp+1];
  ionfraction = pow(10.,(1.-t)*(1.-u)*y1+t*(1.-u)*y2+t*u*y3+(1.-t)*u*y4);
#ifdef VERBOSE  
  fprintf(stdout, "%d: n_h=%f %d, T=%f(%d)  %g %g  %g\n",ionid,log10(n_h),inh,log10(temp),itemp,y1,y3,ionfraction);
#endif  
  return ionfraction;
}

void load_fraction_tables()
/*
  --------------------------------------------------------
  load_fraction_tables: reads in the tables of fractions
  of species (by number relative to the total number 
  of nuclei of the given element) as a function of
  density and temperature. At the present time, the
  following species are included: HI, HeI, HeII, CII, CIV,
  O6, NV, SiIV. UH August 1996 
  --------------------------------------------------------- 
*/
{
  int i;
  FILE *table1,*table2;
  int inh,itemp;
  int z10,zlo,zhi;
  float ftmp;
  float redzlo,redzhi;
  char fname1[80],fname2[80],prefix[50];
  char suffix[12];
  
  sprintf(prefix,"/home/shuiyao/temp/specexbin/ionfiles/");
  sprintf(suffix,"_i9");
  /* sprintf(suffix,"_i31"); */

  z10 = (int) (redshift*10.);
  zlo = z10;
  sprintf(fname1,"%slt%.2d%s%s",prefix,z10,ionbkgd,suffix);
  while ((table1=fopen(fname1,"r")) == NULL && zlo > 0) {
    zlo--;
    sprintf(fname1,"%slt%.2d%s%s",prefix,zlo,ionbkgd,suffix);
  }
  zhi = z10+1;
  sprintf(fname2,"%slt%.2d%s%s",prefix,zhi,ionbkgd,suffix);
  while ((table2=fopen(fname2,"r")) == NULL && zhi < 80) {
    zhi++;
    sprintf(fname2,"%slt%.2d%s%s",prefix,zhi,ionbkgd,suffix);
  }

  if ( table1 == NULL && table2 == NULL ) {
    printf("<unable to open files %s %s>\n",fname1,fname2);
    exit(-1);
  }
  if ( table1 == NULL ) fprintf(stderr,"Using lookup table:%s\n",fname2);
  else if( table2 == NULL ) fprintf(stderr,"Using lookup tables:%s\n",fname1);
  else fprintf(stderr,"Using lookup tables:\n %s\n %s\n",fname1,fname2);
  if( table2 == NULL ) {
    table2 = table1;
    table1 = NULL;
  }
  redzlo = 0.1*zlo;
  redzhi = 0.1*zhi;
#ifdef VERBOSE	
  fprintf(stderr,"Ionfrac: redshift = %5.3f redzlo = %5.3f redzhi = %5.3f\n",redshift,redzlo,redzhi);
  fprintf(stderr,"%g %g %g\n",redshift,redzlo,redzhi);
#endif	

  if( table1 != NULL )
    for(inh=0;inh<NHPTS;inh++) {
      for(itemp=0;itemp<TPTS;itemp++)
	for(i=0;i<nions;i++) fscanf(table1,"%g ", &fractab[i][inh][itemp]);
    }
  if( table1 != NULL ) fclose(table1);

  for(inh=0;inh<NHPTS;inh++) {
    for(itemp=0;itemp<TPTS;itemp++) {
      for(i=0;i<nions;i++) {
	fscanf(table2,"%g ", &ftmp);
	if( table1 != NULL ) fractab[i][inh][itemp] = ((redzhi-redshift)*fractab[i][inh][itemp] + (redshift-redzlo)*ftmp)/(redzhi-redzlo);
	else fractab[i][inh][itemp] = ftmp;
      }
    }
  } 
  fclose(table2);

  /* Initialize gridpoint vectors for n_h and T */
  for(inh=0;inh<NHPTS;inh++) nhl[inh] = NHLOW + inh*DELTANH;
  for(itemp=0;itemp<TPTS;itemp++) tl[itemp] = TLOW + itemp*DELTAT;
	
  return;
}

#endif // IONS, Begining
