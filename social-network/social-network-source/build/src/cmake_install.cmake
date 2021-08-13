# Install script for directory: /social-network-microservices/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local/bin")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/social-network-microservices/build/src/TextService/cmake_install.cmake")
  include("/social-network-microservices/build/src/UniqueIdService/cmake_install.cmake")
  include("/social-network-microservices/build/src/UserService/cmake_install.cmake")
  include("/social-network-microservices/build/src/SocialGraphService/cmake_install.cmake")
  include("/social-network-microservices/build/src/WriteHomeTimelineService/cmake_install.cmake")
  include("/social-network-microservices/build/src/PostStorageSerivce/cmake_install.cmake")
  include("/social-network-microservices/build/src/UserTimelineService/cmake_install.cmake")
  include("/social-network-microservices/build/src/ComposePostService/cmake_install.cmake")
  include("/social-network-microservices/build/src/UserMentionService/cmake_install.cmake")
  include("/social-network-microservices/build/src/UrlShortenService/cmake_install.cmake")
  include("/social-network-microservices/build/src/MediaService/cmake_install.cmake")
  include("/social-network-microservices/build/src/HomeTimelineService/cmake_install.cmake")

endif()

