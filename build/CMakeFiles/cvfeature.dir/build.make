# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/milab/homework/MatchFeatures

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/milab/homework/MatchFeatures/build

# Include any dependencies generated for this target.
include CMakeFiles/cvfeature.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cvfeature.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cvfeature.dir/flags.make

CMakeFiles/cvfeature.dir/main.cpp.o: CMakeFiles/cvfeature.dir/flags.make
CMakeFiles/cvfeature.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/milab/homework/MatchFeatures/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cvfeature.dir/main.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cvfeature.dir/main.cpp.o -c /home/milab/homework/MatchFeatures/main.cpp

CMakeFiles/cvfeature.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cvfeature.dir/main.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/milab/homework/MatchFeatures/main.cpp > CMakeFiles/cvfeature.dir/main.cpp.i

CMakeFiles/cvfeature.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cvfeature.dir/main.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/milab/homework/MatchFeatures/main.cpp -o CMakeFiles/cvfeature.dir/main.cpp.s

CMakeFiles/cvfeature.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/cvfeature.dir/main.cpp.o.requires

CMakeFiles/cvfeature.dir/main.cpp.o.provides: CMakeFiles/cvfeature.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/cvfeature.dir/build.make CMakeFiles/cvfeature.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/cvfeature.dir/main.cpp.o.provides

CMakeFiles/cvfeature.dir/main.cpp.o.provides.build: CMakeFiles/cvfeature.dir/main.cpp.o


# Object files for target cvfeature
cvfeature_OBJECTS = \
"CMakeFiles/cvfeature.dir/main.cpp.o"

# External object files for target cvfeature
cvfeature_EXTERNAL_OBJECTS =

cvfeature: CMakeFiles/cvfeature.dir/main.cpp.o
cvfeature: CMakeFiles/cvfeature.dir/build.make
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: /home/milab/lib/deploy/opencv/lib/libopencv_world.so.4.5.0
cvfeature: CMakeFiles/cvfeature.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/milab/homework/MatchFeatures/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cvfeature"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cvfeature.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cvfeature.dir/build: cvfeature

.PHONY : CMakeFiles/cvfeature.dir/build

CMakeFiles/cvfeature.dir/requires: CMakeFiles/cvfeature.dir/main.cpp.o.requires

.PHONY : CMakeFiles/cvfeature.dir/requires

CMakeFiles/cvfeature.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cvfeature.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cvfeature.dir/clean

CMakeFiles/cvfeature.dir/depend:
	cd /home/milab/homework/MatchFeatures/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/milab/homework/MatchFeatures /home/milab/homework/MatchFeatures /home/milab/homework/MatchFeatures/build /home/milab/homework/MatchFeatures/build /home/milab/homework/MatchFeatures/build/CMakeFiles/cvfeature.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cvfeature.dir/depend

