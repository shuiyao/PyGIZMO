#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include "loadhdf5.h"
#include "gadgetdefs.h"
#include "grid2d.h"

struct NodesStruct *nodes_array;

int ncells_x;
int ncells_y;
char infilename[200], outfilename[200];

int load_grid(int nx, int ny)
{
  int i, ix, iy;
  float dx, dy, x;
  int ncells;

  ncells = nx * ny;
  nodes_array = (struct NodesStruct *)
    malloc(ncells*sizeof(*nodes_array));
  dx = (XMAX - XMIN) / nx;
  dy = (YMAX - YMIN) / ny;
  for(ix=0;ix<nx;ix++) {
    x = XMIN + dx * ix;
    for (iy=0;iy<ny;iy++){
      i = ix * ny + iy;
      nodes_array[i].x = x;
      nodes_array[i].y = YMIN + dy * iy;
      nodes_array[i].z = 0;
      nodes_array[i].m = 0.0;
      nodes_array[i].met = 0.0;      
    }
  }
  printf("Grid Loaded.\n");
  return 1;
}

void GetFilenames(char *snapbase, char *outputbase, int snapnum)
{
  char snapstr[4];
  get_snap_string(snapnum, snapstr);
#ifdef IONS
  sprintf(outfilename, "%s/tabion_%s.csv", outputbase, snapstr);
#else  
  sprintf(outfilename, "%s/tabmet_%s.csv", outputbase, snapstr);
#endif
  sprintf(infilename, "%s/snapshot_%s", snapbase, snapstr);
  fprintf(stdout, "Infile: %s\n", infilename);
}

int grid_count(float x, float y, float m, float meto)
{
  int ix, iy, i;
  ix = (int)((x - XMIN) / (XMAX - XMIN) * ncells_x);
  if (ix < ncells_x && ix > 0){
    iy = (int)((y - YMIN) / (YMAX - YMIN) * ncells_y);
    if (iy < ncells_y && iy > 0){
      i = ix * ncells_y + iy;
      nodes_array[i].z ++;
      nodes_array[i].m += m;
      nodes_array[i].met += m * meto / 8.65e-3 * 0.0122;
#ifdef IONS
      nodes_array[i].mhi += m * XH * fHI;
      nodes_array[i].mciv += m * metc * fCIV;      
      nodes_array[i].movi += m * meto * fOVI;
      nodes_array[i].mneviii += m * meto / 8.65e-3 * 2.22e-3 * fNeVIII;      
      nodes_array[i].msiiv += m * meto / 8.65e-3 * 1.08e-3 * fSiIV;
#endif	
      return 0;
    }
  }
  return 1;
}

void ReadData(char *filename)
{
  int i, Nout = 0, Ntot = 0;
  double LogRho, LogT;
  double rho;
  double Mass;
  double MeanWeight;

  load_hdf5(filename);
  cosmounits();

#ifdef IONS
  unit_Density = 1.87e-29 * gheader.HubbleParam * gheader.HubbleParam;
  a3inv = 1. / (gheader.time * gheader.time * gheader.time);
  InitIons(1./header.time - 1.);  // Including load_fraction_tables()
  /* fHI = IonFrac(1.e3, 10.0*MHYDR, 0); // test */
#endif  

  for(i=0;i<gheader.npart[0];i++) 
    {
      rho = P[i].Rho * UNIT_M / pow(UNIT_L, 3) * gheader.HubbleParam * gheader.HubbleParam; // density in c.g.s units
      LogRho = log10(rho / unit_Density / OMEGABARYON);
      MeanWeight = (1 + 4 * XHE) / (1 + P[i].Ne + XHE);
      LogT = P[i].Temp * unit_Temp;
      LogT *= GAMMA_MINUS1 * PROTONMASS / BOLTZMANN * MeanWeight;
#ifdef IONS
      fHI = 0.0;
      fCIV = 0.0;      
      fOVI = 0.0;
      fNeVIII = 0.0;
      fSiIV = 0.0;
      rho *= a3inv;
      fHI = IonFrac(LogT, rho, 0); // Global Variable that enters grid_count
      fCIV = IonFrac(LogT, rho, 3);
      fOVI = IonFrac(LogT, rho, 5);
      fNeVIII = IonFrac(LogT, rho, 6);      
      fSiIV = IonFrac(LogT, rho, 8);
      /* HI, HeII, CIII, CIV, OIV, OVI, NeVIII, MgII, SiIV */
#endif
      LogT = log10(LogT);
      Mass = P[i].Mass;
      Nout += grid_count(LogRho, LogT, Mass, P[i].metal[4]);
      Ntot ++;
    }
  printf("Data Reading Done, Nout/Ntot = %d/%d\n", Nout, Ntot);
}

void WriteGrid(char *filename)
{
  FILE *outfile;
  int ix, iy, i;
  outfile = fopen(filename, "w");
  fprintf(outfile, "LogRho,LogT,count,Mass,Zmet");
#ifdef IONS
  fprintf(outfile, ",MHI,MCIV,MOVI,MNeVIII,MSiIV");
#endif
  fprintf(outfile, "\n");  
  for(ix=0;ix<ncells_x;ix++) {
    for (iy=0;iy<ncells_y;iy++){
      i = ix * ncells_y + iy;
      fprintf(outfile, "%g,%g,%d,%g,%g",
	      nodes_array[i].x,
	      nodes_array[i].y,
	      nodes_array[i].z,
	      nodes_array[i].m,
	      nodes_array[i].met);      
#ifdef IONS
      fprintf(outfile, ",%g,%g,%g,%g,%g\n",
	      nodes_array[i].mhi, nodes_array[i].mciv,
	      nodes_array[i].movi, nodes_array[i].mneviii,
	      nodes_array[i].msiiv);
#else
      fprintf(outfile, "\n");       
#endif      
    }
  }
  fclose(outfile);
}

int build_phase_diagram(char *snapbase, char *outputbase, int snapnum, int nx, int ny)
{
  ncells_x = nx;
  ncells_y = ny;
  fprintf(stdout, "Generating %d x %d grid.\n", ncells_x, ncells_y);  
  fprintf(stdout, "snapbase: %s\n", snapbase);
  GetFilenames(snapbase, outputbase, snapnum);
  load_grid(ncells_x, ncells_y);
  ReadData(infilename);
  WriteGrid(outfilename);
  free(nodes_array);
  return 1;
}
