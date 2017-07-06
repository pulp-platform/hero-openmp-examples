ifndef PULP_SDK_HOME
$(error PULP_SDK_HOME is not set)
endif

ifndef_any_of = $(filter undefined,$(foreach v,$(1),$(origin $(v))))

############## Name of output executable file
ifndef EXE
EXE=app.exe
endif

EXT_DEF=
############## Default compiler and possible option sets
CROSS_COMPILER=arm-linux-gnueabihf
CC      = $(CROSS_COMPILER)-gcc
LD      = $(CROSS_COMPILER)-ld
AS      = $(CROSS_COMPILER)-gcc
OBJCOPY = $(CROSS_COMPILER)-objcopy
OBJDUMP = $(CROSS_COMPILER)-objdump
OPT     =-O2 -g3 -fopenmp -fdump-tree-optimized 

############## Includes
INCDIR        += -I. -I$(PULP_SDK_HOME)/install/arm/include -I${PULP_SDK_HOME}/install/include

PLP_CFLAGS=-O3 -DPULP -D__PULP__ -DBENCH=1 -DFP_SW_EMUL -DOR10N5 -D__OR10N5__ -DOR1K -D__OR1K__ -D__LIBGOMP__ -DL2_SIZE=262144 -DL1_SIZE=262144 -DNB_PE=8 -DSTACK_SIZE=1024 -DNB_CLUSTER=1 -DPLP_NO_BUILTIN -DPULP_CHIP=CHIP_BIGPULP_Z_7045_O -DPULP_CHIP_FAMILY=CHIP_BIGPULP -DPULP_CHIP_VERSION=0 -DPULP_CORE=CORE_OR1K_V5

COMMON_CFLAGS += ${INCDIR} $(PLP_CFLAGS)
CFLAGS        += $(OPT) -Wall $(COMMON_CFLAGS) ${EXT_DEF} -fpulp-rab -foffload="$(PLP_CFLAGS)"
ASFLAGS       += $(OPT) $(COMMON_CFLAGS)
LDFLAGS       += -L$(PULP_SDK_HOME)/install/arm/lib -lpulp-api

############################ OBJECTS ###################################
COBJS  = $(CSRCS:.c=.o)
ASOBJS = $(ASSRCS:.S=.o)
OBJS = $(COBJS) $(ASOBJS)

all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $@

.PHONY: clean prepare run
clean:
	rm -rf *.o *~  $(EXE) $(OBJS) $(CSRCS).*

soft-reset-board:
ifndef PULP_EMU_ADDR
$(error PULP_EMU_ADDR is not set)
endif
	ssh -t $(PULP_EMU_ADDR) 'sudo init_bigpulp.sh'

prepare: $(EXE)
ifndef PULP_EMU_ADDR
$(error PULP_EMU_ADDR is not set)
endif
	scp $(EXE) $(PULP_EMU_ADDR):~

run: prepare $(EXE)
ifeq ($(call ifndef_any_of,PULP_EMU_ADDR PULP_EMU_SHARE_DIR),)
	ssh -t $(PULP_EMU_ADDR) 'export LD_LIBRARY_PATH='"'$(PULP_EMU_SHARE_DIR)/lib'":"'$(PULP_EMU_SHARE_DIR)/arm-linux-gnueabihf/lib'"'; ./$(EXE) $(RUN_ARGS)'
else
$(error PULP_EMU_ADDR and/or PULP_EMU_SHARE_DIR is not set)
endif
