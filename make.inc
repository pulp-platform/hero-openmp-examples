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
CC      = $(CROSS_COMPILE)gcc
LD      = $(CROSS_COMPILE)ld
AS      = $(CROSS_COMPILE)gcc
OBJCOPY = $(CROSS_COMPILE)objcopy
OBJDUMP = $(CROSS_COMPILE)objdump
OPT     =-O2 -g3 -fopenmp

############## Includes
INCDIR        += -I.
COMMON_CFLAGS += ${INCDIR} -v
CFLAGS        += $(OPT) -Wall $(COMMON_CFLAGS) ${EXT_DEF}
ASFLAGS       += $(OPT) $(COMMON_CFLAGS)
LDFLAGS       +=

############################ OBJECTS ###################################
COBJS  = $(CSRCS:.c=.o)
ASOBJS = $(ASSRCS:.S=.o)
OBJS = $(COBJS) $(ASOBJS)

all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $@

.PHONY: clean prepare run
clean:
	rm -rf *.o *~  $(EXE) $(OBJS) $(EXE).* offload.bin offload.so

soft-reset-board:
ifndef PULP_EMU_ADDR
$(error PULP_EMU_ADDR is not set)
endif
	ssh -t $(PULP_EMU_ADDR) 'sudo /hsa/bin/boot-hero'

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
