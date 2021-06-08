OPT     = -DH5_USE_16_API

# Default parameters for the 2D binning
OPT    += -D_GRID2D
OPT    += -DNUMOFNODES_X=256    # number of cells (x) 
OPT    += -DNUMOFNODES_Y=256    # number of cells (y) 
OPT    += -DXMIN=-1.5           # Lower limit (x), log(rho/rho_crit) 
OPT    += -DXMAX=7.0            # Upper limit (x), log(rho/rho_crit) 
OPT    += -DYMIN=3.0            # Lower limit (y), log(T/K)
OPT    += -DYMAX=8.0            # Upper limit (y), log(T/K)
OPT    += -DOMEGABARYON=0.045   # Baryon fraction of the Universe
#OPT    += -DIONS                # Compute ion fractions or not

CC 	= gcc
CFLAGS  = -Wall -Wextra -Wno-unused-parameter
CFLAGS 	+= -O3
CFLAGS  += $(OPT)
LDFLAGS = -lm -lhdf5 -lz

LIB	= cpygizmo.so
SRCDIR	= src
INCLDIR	= include
BUILDIR	= build


# derived variables
SRC	= $(wildcard $(SRCDIR)/*.c)
OBJ 	= $(addprefix $(BUILDIR)/,$(notdir $(SRC:%.c=%.o)))
HEADERS = $(wildcard $(INCLDIR)/*.h)

.PHONY:	all clean
.SECONDARY: main-build

all: $(LIB)

# also let depend on headers to enshure proper rebuilds
$(OBJ) : $(BUILDIR)/%.o : $(SRCDIR)/%.c $(HEADERS) Makefile
	@mkdir -p $(BUILDIR)
	$(CC) -fPIC $(CFLAGS) $(IPATH) -I./$(INCLDIR) -c $< -o $@

$(LIB): $(OBJ) Makefile
	$(CC) -fPIC -shared $(LPATH) $(OBJ) $(LDFLAGS) -o $@

clean:
	rm -r $(BUILDIR)
	rm $(LIB)

