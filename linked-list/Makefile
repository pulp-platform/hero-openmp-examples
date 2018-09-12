CSRCS = linked-list.c
-include $(PWD)/../make.inc

prepare::
ifndef HERO_TARGET_HOST
$(error HERO_TARGET_HOST is not set)
endif
	scp *.txt $(HERO_TARGET_HOST):$(HERO_TARGET_PATH_APPS)/.
