# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/d/PKU/projects/pyabacus_diago/diago

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/d/PKU/projects/pyabacus_diago/diago/build

# Include any dependencies generated for this target.
include CMakeFiles/diago.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/diago.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/diago.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/diago.dir/flags.make

CMakeFiles/diago.dir/diagh_consts.cpp.o: CMakeFiles/diago.dir/flags.make
CMakeFiles/diago.dir/diagh_consts.cpp.o: ../diagh_consts.cpp
CMakeFiles/diago.dir/diagh_consts.cpp.o: CMakeFiles/diago.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/PKU/projects/pyabacus_diago/diago/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/diago.dir/diagh_consts.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/diago.dir/diagh_consts.cpp.o -MF CMakeFiles/diago.dir/diagh_consts.cpp.o.d -o CMakeFiles/diago.dir/diagh_consts.cpp.o -c /mnt/d/PKU/projects/pyabacus_diago/diago/diagh_consts.cpp

CMakeFiles/diago.dir/diagh_consts.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/diago.dir/diagh_consts.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/PKU/projects/pyabacus_diago/diago/diagh_consts.cpp > CMakeFiles/diago.dir/diagh_consts.cpp.i

CMakeFiles/diago.dir/diagh_consts.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/diago.dir/diagh_consts.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/PKU/projects/pyabacus_diago/diago/diagh_consts.cpp -o CMakeFiles/diago.dir/diagh_consts.cpp.s

CMakeFiles/diago.dir/diago_dav_subspace.cpp.o: CMakeFiles/diago.dir/flags.make
CMakeFiles/diago.dir/diago_dav_subspace.cpp.o: ../diago_dav_subspace.cpp
CMakeFiles/diago.dir/diago_dav_subspace.cpp.o: CMakeFiles/diago.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/PKU/projects/pyabacus_diago/diago/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/diago.dir/diago_dav_subspace.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/diago.dir/diago_dav_subspace.cpp.o -MF CMakeFiles/diago.dir/diago_dav_subspace.cpp.o.d -o CMakeFiles/diago.dir/diago_dav_subspace.cpp.o -c /mnt/d/PKU/projects/pyabacus_diago/diago/diago_dav_subspace.cpp

CMakeFiles/diago.dir/diago_dav_subspace.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/diago.dir/diago_dav_subspace.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/PKU/projects/pyabacus_diago/diago/diago_dav_subspace.cpp > CMakeFiles/diago.dir/diago_dav_subspace.cpp.i

CMakeFiles/diago.dir/diago_dav_subspace.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/diago.dir/diago_dav_subspace.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/PKU/projects/pyabacus_diago/diago/diago_dav_subspace.cpp -o CMakeFiles/diago.dir/diago_dav_subspace.cpp.s

CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.o: CMakeFiles/diago.dir/flags.make
CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.o: ../module_base/parallel_reduce.cpp
CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.o: CMakeFiles/diago.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/PKU/projects/pyabacus_diago/diago/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.o -MF CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.o.d -o CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.o -c /mnt/d/PKU/projects/pyabacus_diago/diago/module_base/parallel_reduce.cpp

CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/PKU/projects/pyabacus_diago/diago/module_base/parallel_reduce.cpp > CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.i

CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/PKU/projects/pyabacus_diago/diago/module_base/parallel_reduce.cpp -o CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.s

CMakeFiles/diago.dir/module_base/module_device/device.cpp.o: CMakeFiles/diago.dir/flags.make
CMakeFiles/diago.dir/module_base/module_device/device.cpp.o: ../module_base/module_device/device.cpp
CMakeFiles/diago.dir/module_base/module_device/device.cpp.o: CMakeFiles/diago.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/PKU/projects/pyabacus_diago/diago/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/diago.dir/module_base/module_device/device.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/diago.dir/module_base/module_device/device.cpp.o -MF CMakeFiles/diago.dir/module_base/module_device/device.cpp.o.d -o CMakeFiles/diago.dir/module_base/module_device/device.cpp.o -c /mnt/d/PKU/projects/pyabacus_diago/diago/module_base/module_device/device.cpp

CMakeFiles/diago.dir/module_base/module_device/device.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/diago.dir/module_base/module_device/device.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/PKU/projects/pyabacus_diago/diago/module_base/module_device/device.cpp > CMakeFiles/diago.dir/module_base/module_device/device.cpp.i

CMakeFiles/diago.dir/module_base/module_device/device.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/diago.dir/module_base/module_device/device.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/PKU/projects/pyabacus_diago/diago/module_base/module_device/device.cpp -o CMakeFiles/diago.dir/module_base/module_device/device.cpp.s

CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.o: CMakeFiles/diago.dir/flags.make
CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.o: ../module_base/module_device/memory_op.cpp
CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.o: CMakeFiles/diago.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/PKU/projects/pyabacus_diago/diago/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.o -MF CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.o.d -o CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.o -c /mnt/d/PKU/projects/pyabacus_diago/diago/module_base/module_device/memory_op.cpp

CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/PKU/projects/pyabacus_diago/diago/module_base/module_device/memory_op.cpp > CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.i

CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/PKU/projects/pyabacus_diago/diago/module_base/module_device/memory_op.cpp -o CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.s

CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.o: CMakeFiles/diago.dir/flags.make
CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.o: ../module_hsolver/kernels/dngvd_op.cpp
CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.o: CMakeFiles/diago.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/PKU/projects/pyabacus_diago/diago/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.o -MF CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.o.d -o CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.o -c /mnt/d/PKU/projects/pyabacus_diago/diago/module_hsolver/kernels/dngvd_op.cpp

CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/PKU/projects/pyabacus_diago/diago/module_hsolver/kernels/dngvd_op.cpp > CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.i

CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/PKU/projects/pyabacus_diago/diago/module_hsolver/kernels/dngvd_op.cpp -o CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.s

CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.o: CMakeFiles/diago.dir/flags.make
CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.o: ../module_hsolver/kernels/math_kernel_op.cpp
CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.o: CMakeFiles/diago.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/PKU/projects/pyabacus_diago/diago/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.o -MF CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.o.d -o CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.o -c /mnt/d/PKU/projects/pyabacus_diago/diago/module_hsolver/kernels/math_kernel_op.cpp

CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/PKU/projects/pyabacus_diago/diago/module_hsolver/kernels/math_kernel_op.cpp > CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.i

CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/PKU/projects/pyabacus_diago/diago/module_hsolver/kernels/math_kernel_op.cpp -o CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.s

# Object files for target diago
diago_OBJECTS = \
"CMakeFiles/diago.dir/diagh_consts.cpp.o" \
"CMakeFiles/diago.dir/diago_dav_subspace.cpp.o" \
"CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.o" \
"CMakeFiles/diago.dir/module_base/module_device/device.cpp.o" \
"CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.o" \
"CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.o" \
"CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.o"

# External object files for target diago
diago_EXTERNAL_OBJECTS =

libdiago.so: CMakeFiles/diago.dir/diagh_consts.cpp.o
libdiago.so: CMakeFiles/diago.dir/diago_dav_subspace.cpp.o
libdiago.so: CMakeFiles/diago.dir/module_base/parallel_reduce.cpp.o
libdiago.so: CMakeFiles/diago.dir/module_base/module_device/device.cpp.o
libdiago.so: CMakeFiles/diago.dir/module_base/module_device/memory_op.cpp.o
libdiago.so: CMakeFiles/diago.dir/module_hsolver/kernels/dngvd_op.cpp.o
libdiago.so: CMakeFiles/diago.dir/module_hsolver/kernels/math_kernel_op.cpp.o
libdiago.so: CMakeFiles/diago.dir/build.make
libdiago.so: /usr/lib/x86_64-linux-gnu/libopenblas.so
libdiago.so: CMakeFiles/diago.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/PKU/projects/pyabacus_diago/diago/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX shared library libdiago.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/diago.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/diago.dir/build: libdiago.so
.PHONY : CMakeFiles/diago.dir/build

CMakeFiles/diago.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/diago.dir/cmake_clean.cmake
.PHONY : CMakeFiles/diago.dir/clean

CMakeFiles/diago.dir/depend:
	cd /mnt/d/PKU/projects/pyabacus_diago/diago/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/PKU/projects/pyabacus_diago/diago /mnt/d/PKU/projects/pyabacus_diago/diago /mnt/d/PKU/projects/pyabacus_diago/diago/build /mnt/d/PKU/projects/pyabacus_diago/diago/build /mnt/d/PKU/projects/pyabacus_diago/diago/build/CMakeFiles/diago.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/diago.dir/depend

