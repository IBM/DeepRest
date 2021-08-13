add_executable(
    UserTimelineService
    UserTimelineService.cpp
    ${THRIFT_GEN_CPP_DIR}/UserTimelineService.cpp
    ${THRIFT_GEN_CPP_DIR}/social_network_types.cpp
    ${THRIFT_GEN_CPP_DIR}/PostStorageService.cpp
)

target_include_directories(
    UserTimelineService PRIVATE
    ${MONGOC_INCLUDE_DIRS}
    /usr/local/include/jaegertracing
    /usr/local/include/cpp_redis
    /usr/include
)

target_link_libraries(
    UserTimelineService
    ${MONGOC_LIBRARIES}
    nlohmann_json::nlohmann_json
    ${THRIFT_LIB}
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    Boost::log
    Boost::log_setup
    jaegertracing
    /usr/local/lib/libcpp_redis.a
    /usr/local/lib/libtacopie.a
    /usr/local/lib/libthrift.so
    /usr/local/lib/libthriftnb.so
)

install(TARGETS UserTimelineService DESTINATION ./)