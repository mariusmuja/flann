# Enforce NEWLINE_STYLE to UNIX to prevent Git from finding the file dirty.
configure_file( setup.py.tpl setup.py NEWLINE_STYLE UNIX )

install( DIRECTORY pyflann DESTINATION share/flann/python )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/setup.py DESTINATION share/flann/python )


# python instalation
if (PYTHON_EXECUTABLE)
    install(CODE "execute_process(
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py install
        WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\")")
endif()
