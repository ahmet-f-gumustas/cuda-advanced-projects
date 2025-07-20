# Examples CMakeLists.txt

# ResNet inference example
add_executable(resnet_inference resnet_inference.cpp)
target_link_libraries(resnet_inference deep_engine ${OpenCV_LIBS})

# YOLO inference example  
add_executable(yolo_inference yolo_inference.cpp)
target_link_libraries(yolo_inference deep_engine ${OpenCV_LIBS})

# Install examples
install(TARGETS resnet_inference yolo_inference
    RUNTIME DESTINATION bin/examples
)

# Copy example data
install(FILES 
    imagenet_classes.txt
    coco_classes.txt
    DESTINATION share/deep_engine/examples
)