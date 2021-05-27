void get_snap_string(int snapnum, char *snapstr);
void read_header(struct tipsy_header *head, FILE *ftipsy);
void headerinfo(struct tipsy_header *header);
#ifdef IONS
int InitIons();
double IonFrac(float temp, float density, int ionid);
void load_fraction_tables();
#endif
