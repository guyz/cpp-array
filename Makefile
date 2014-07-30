###############################################################################################
#
#
#
#     _/_/_/  _/_/_/    _/_/_/                  _/_/_/  _/  _/_/  _/  _/_/    _/_/_/  _/    _/
#  _/        _/    _/  _/    _/  _/_/_/_/_/  _/    _/  _/_/      _/_/      _/    _/  _/    _/
# _/        _/    _/  _/    _/              _/    _/  _/        _/        _/    _/  _/    _/
#  _/_/_/  _/_/_/    _/_/_/                  _/_/_/  _/        _/          _/_/_/    _/_/_/
#         _/        _/                                                                  _/
#        _/        _/                                                              _/_/
#
###############################################################################################


# Configuration options.
cxx        = not-set
CXX        = not-set
fc         = not-set
FC         = not-set
prefix     = not-set
doc        = not-set
latex      = not-set
build      = not-set
cuda       = not-set
dev        = not-set


# Basically proxies everything to the builddir cmake.

cputype = $(shell uname -m | sed "s/\\ /_/g")
systype = $(shell uname -s)

BUILDDIR = build/$(systype)-$(cputype)

# Process configuration options.
CONFIG_FLAGS = -DCMAKE_VERBOSE_MAKEFILE=0
ifneq ($(prefix), not-set)
    CONFIG_FLAGS += -DCMAKE_INSTALL_PREFIX=$(prefix)
endif
ifneq ($(cxx), not-set)
    CONFIG_FLAGS += -DCMAKE_CXX_COMPILER=$(cxx)
endif
ifneq ($(CXX), not-set)
    CONFIG_FLAGS += -DCMAKE_CXX_COMPILER=$(CXX)
endif
ifneq ($(fc), not-set)
    CONFIG_FLAGS += -DCMAKE_Fortran_COMPILER=$(fc)
endif
ifneq ($(FC), not-set)
    CONFIG_FLAGS += -DCMAKE_Fortran_COMPILER=$(FC)
endif
ifneq ($(doc), not-set)
    CONFIG_FLAGS += -DCPP-ARRAY_DOCUMENTATION=$(doc)
endif
ifneq ($(latex), not-set)
    CONFIG_FLAGS += -DLaTeX=$(latex)
endif
ifneq ($(build), not-set)
    CONFIG_FLAGS += -DCMAKE_BUILD_TYPE=$(build)
endif
ifneq ($(cuda), not-set)
    CONFIG_FLAGS += -DCUDA=$(cuda)
endif
ifneq ($(dev), not-set)
    CONFIG_FLAGS += -DCPP-ARRAY_DEV=$(dev)
endif



define run-config
mkdir -p $(BUILDDIR)
cd $(BUILDDIR) && cmake $(CURDIR) $(CONFIG_FLAGS)
endef

all clean install doc check examples package package_source depend edit_cache install/local install/strip list_install_components rebuild_cache a.out:
	@if [ ! -f $(BUILDDIR)/Makefile ]; then \
		more README; \
	else \
	  	make -C $(BUILDDIR) $@ $(MAKEFLAGS); \
	fi

test:
	@if [ ! -f $(BUILDDIR)/Makefile ]; then \
		more README; \
	else \
	  	make -C $(BUILDDIR) check $(MAKEFLAGS); \
	fi

uninstall:
	xargs rm < $(BUILDDIR)/install_manifest.txt

config: distclean
	$(run-config)

distclean:
	rm -rf $(BUILDDIR)

remake:
	find . -name CMakeLists.txt -exec touch {} ';'

dist:
	utils/mkdist.sh $(PKGNAME)

.PHONY: config distclean all clean install uninstall remake dist doc check examples package package_source depend edit_cache install/local install/strip list_install_components rebuild_cache test a.out