# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/astlesylvesterdsa/Projects/hnlx/src/C++

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/astlesylvesterdsa/Projects/hnlx/src/C++/build

# Include any dependencies generated for this target.
include CMakeFiles/hnlx.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/hnlx.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/hnlx.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hnlx.dir/flags.make

CMakeFiles/hnlx.dir/codegen:
.PHONY : CMakeFiles/hnlx.dir/codegen

CMakeFiles/hnlx.dir/hnlx.cpp.o: CMakeFiles/hnlx.dir/flags.make
CMakeFiles/hnlx.dir/hnlx.cpp.o: /Users/astlesylvesterdsa/Projects/hnlx/src/C++/hnlx.cpp
CMakeFiles/hnlx.dir/hnlx.cpp.o: CMakeFiles/hnlx.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/astlesylvesterdsa/Projects/hnlx/src/C++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hnlx.dir/hnlx.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/hnlx.dir/hnlx.cpp.o -MF CMakeFiles/hnlx.dir/hnlx.cpp.o.d -o CMakeFiles/hnlx.dir/hnlx.cpp.o -c /Users/astlesylvesterdsa/Projects/hnlx/src/C++/hnlx.cpp

CMakeFiles/hnlx.dir/hnlx.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/hnlx.dir/hnlx.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/astlesylvesterdsa/Projects/hnlx/src/C++/hnlx.cpp > CMakeFiles/hnlx.dir/hnlx.cpp.i

CMakeFiles/hnlx.dir/hnlx.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/hnlx.dir/hnlx.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/astlesylvesterdsa/Projects/hnlx/src/C++/hnlx.cpp -o CMakeFiles/hnlx.dir/hnlx.cpp.s

# Object files for target hnlx
hnlx_OBJECTS = \
"CMakeFiles/hnlx.dir/hnlx.cpp.o"

# External object files for target hnlx
hnlx_EXTERNAL_OBJECTS =

libhnlx.a: CMakeFiles/hnlx.dir/hnlx.cpp.o
libhnlx.a: CMakeFiles/hnlx.dir/build.make
libhnlx.a: CMakeFiles/hnlx.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/astlesylvesterdsa/Projects/hnlx/src/C++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libhnlx.a"
	$(CMAKE_COMMAND) -P CMakeFiles/hnlx.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hnlx.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hnlx.dir/build: libhnlx.a
.PHONY : CMakeFiles/hnlx.dir/build

CMakeFiles/hnlx.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hnlx.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hnlx.dir/clean

CMakeFiles/hnlx.dir/depend:
	cd /Users/astlesylvesterdsa/Projects/hnlx/src/C++/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/astlesylvesterdsa/Projects/hnlx/src/C++ /Users/astlesylvesterdsa/Projects/hnlx/src/C++ /Users/astlesylvesterdsa/Projects/hnlx/src/C++/build /Users/astlesylvesterdsa/Projects/hnlx/src/C++/build /Users/astlesylvesterdsa/Projects/hnlx/src/C++/build/CMakeFiles/hnlx.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/hnlx.dir/depend

