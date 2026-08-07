.PHONY: all install_main_env install_vt_env remove_main_env remove_vt_env

main_environment_filename ?= environment.yml
vt_environment_filename ?= environment_vt.yml
vt_pip_requirements_filename ?= pip_requirements_vt.txt

main_environment_name := $(shell grep "^name:" \
  $(main_environment_filename) | sed "s/^name: *//")

vt_environment_name := $(shell grep "^name:" \
  $(vt_environment_filename) | sed "s/^name: *//")


all:
	install_main_env install_vt_env

install_main_env:
	conda env create -f $(main_environment_filename)
	
remove_main_env:
	conda env remove -n $(main_environment_name) || true
	
install_vt_env:
	conda env create -f $(vt_environment_filename)
	
remove_vt_env:
	conda env remove -n $(vt_environment_name) || true
