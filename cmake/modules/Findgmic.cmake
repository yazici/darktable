# - Find the native libgmic includes and library
#
# This module defines
#  GMIC_INCLUDE_DIR, where to find gmic.h, etc.
#  GMIC_LIBRARIES, the libraries to link against to use libgmic.
#  GMIC_FOUND, If false, do not try to use libgmic.
# also defined, but not for general use are
#  GMIC_LIBRARY, where to find the libgmic library.


#=============================================================================
# Copyright 2010 henrik andersson
#=============================================================================

include(LibFindMacros)

SET(GMIC_FIND_REQUIRED ${GMIC_FIND_REQUIRED})

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(GMIC_PKGCONF libgmic_qt)

find_path(GMIC_INCLUDE_DIR NAMES gmic_qt_lib.h
  HINTS ${GMIC_PKGCONF_INCLUDE_DIRS}
  /usr/include/
  ENV GMIC_INCLUDE_DIR)
mark_as_advanced(GMIC_INCLUDE_DIR)

set(GMIC_NAMES ${GMIC_NAMES} libgmic_qt libgmic_qt.so)
find_library(GMIC_LIBRARY NAMES ${GMIC_NAMES} 
	HINTS ENV GMIC_LIB_DIR)
mark_as_advanced(GMIC_LIBRARY)

# handle the QUIETLY and REQUIRED arguments and set GMIC_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMIC DEFAULT_MSG GMIC_LIBRARY GMIC_INCLUDE_DIR)

IF(GMIC_FOUND)
  SET(GMIC_LIBRARIES ${GMIC_LIBRARY})
  SET(GMIC_INCLUDE_DIRS ${GMIC_INCLUDE_DIR})
ENDIF(GMIC_FOUND)
