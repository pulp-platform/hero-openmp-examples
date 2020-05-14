# Copyright 2018 (C) ETH Zurich and University of Bologna
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CSRCS=src/main.c src/sobel.c src/file_operations.c
EXE=sobel
CFLAGS=-foffload="-lm"
LDFLAGS=-lm

IMG_DIR      = imgs
IMG_DIR_OUT  = imgs_out
IMAGE_NAME   = img

RUN_ARGS=input.rgb output.gray 192x192 -g output_gray.gray -i output_h.gray output_v.gray

-include $(PWD)/../common/default.mk

clean::
	rm -rf *.gray output*

run: prepare $(EXE)
ifeq ($(call ifndef_any_of,HERO_TARGET_HOST HERO_TARGET_PATH_LIB),)
	# Convert input PNG to RGB and copy it to the board.
	convert input.png input.rgb
	scp input.rgb $(HERO_TARGET_HOST):${exedir}/
	# Launch application on board.
	ssh -t $(HERO_TARGET_HOST) 'export LD_LIBRARY_PATH='"'$(HERO_TARGET_PATH_LIB)'"'; cd ${exedir}; ./$(EXE) $(RUN_ARGS)'
	# Copy result files back.
	scp '$(HERO_TARGET_HOST):${exedir}/*.gray' .
	# Remove temporary directory on board.
	ssh $(HERO_TARGET_HOST) 'rm -rf ${exedir}'
	# Convert all gray files to PNGs.
	/usr/bin/env bash -c 'for f in *.gray; do convert -size 192x192 -depth 8 $$f $$(basename -s .gray $$f).png; done'
else
$(error HERO_TARGET_HOST and/or HERO_TARGET_PATH_LIB is not set)
endif
