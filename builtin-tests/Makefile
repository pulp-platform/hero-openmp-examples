CSRCS   = test.c
CFLAGS += -foffload="-v -fdump-tree-all -flto"

clean::
	rm -rf  test.c.* /tmp/test.o.*
-include ${HERO_OMP_EXAMPLES_DIR}/common/default.mk
