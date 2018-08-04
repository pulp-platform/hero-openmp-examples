CSRCS=src/main.c src/sobel.c src/file_operations.c
EXE=sobel
CFLAGS=-foffload="-lm"
LDFLAGS=-lm

IMG_DIR      = imgs
IMG_DIR_OUT  = imgs_out
IMAGE_NAME   = img

RUN_ARGS=$(IMG_DIR)/$(IMAGE_NAME).rgb $(IMG_DIR)/$(IMAGE_NAME).gray 512x512 -g $(IMG_DIR)/$(IMAGE_NAME)_gray.gray -i $(IMG_DIR)/$(IMAGE_NAME)_h.gray $(IMG_DIR)/$(IMAGE_NAME)_v.gray

-include $(PWD)/../make.inc

copyout:
	convert $(IMG_DIR)/$(IMAGE_NAME).png $(IMG_DIR)/$(IMAGE_NAME).rgb
	rsync -rctacvzP $(IMG_DIR) $(PULP_EMU_ADDR):~

copyin:
	rsync -rctacvzP  $(PULP_EMU_ADDR):~/$(IMG_DIR)/* $(IMG_DIR_OUT)
	convert -size 512x512 -depth 8 $(IMG_DIR_OUT)/$(IMAGE_NAME).gray $(IMG_DIR_OUT)/$(IMAGE_NAME).png
	convert -size 512x512 -depth 8 $(IMG_DIR_OUT)/$(IMAGE_NAME)_gray.gray $(IMG_DIR_OUT)/$(IMAGE_NAME)_gray.png
	convert -size 512x512 -depth 8 $(IMG_DIR_OUT)/$(IMAGE_NAME)_h.gray $(IMG_DIR_OUT)/$(IMAGE_NAME)_h.png
	convert -size 512x512 -depth 8 $(IMG_DIR_OUT)/$(IMAGE_NAME)_v.gray $(IMG_DIR_OUT)/$(IMAGE_NAME)_v.png

remove-output:
	rm -rf $(IMG_DIR_OUT)

test: copyout run copyin
