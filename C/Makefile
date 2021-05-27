CC 	= gcc
CFLAGS  = -Wall -Wextra -Wno-unused-parameter
CFLAGS 	+= -O3
LDFLAGS = -lm

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

