#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point



class PlaneSegmentationNode:
    def __init__(self):
        rospy.init_node("plane_segmentation_node")

        # Create a ROS publisher for the segmented planes (PointCloud2 message)
        self.planes_publisher = rospy.Publisher("/planes", PointCloud2, queue_size=10)

        # Create a ROS publisher for the segmented planes (PointCloud visualization)
        self.planes_visual_publisher = rospy.Publisher("/planes_visual", MarkerArray, queue_size=10)

        # Create a CV Bridge object
        self.bridge = CvBridge()

        # Set camera intrinsics parameters
        self.fx = 575.8157
        self.fy = 575.8157
        self.cx = 319.5
        self.cy = 239.5

        # Initialize the RealSense camera subscriber
        self.color_subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.camera_callback)
        self.depth_subscriber = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)

        # Initialize the depth Image
        self.depth_image = None

    def camera_callback(self, image_msg):
        # Convert the ROS image message to a color frame
        color_frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        # Check if the depth image is available
        if self.depth_image is not None:
            color_image = o3d.geometry.Image(np.array(color_frame))
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, self.depth_image,
                                                                            depth_scale=1.0,
                                                                            convert_rgb_to_intensity=False)

            # Check if the RGBD image has enough points for segmentation
            if rgbd_image.color is None or rgbd_image.depth is None:
                rospy.logwarn("Insufficient points in the point cloud for plane segmentation.")
                return

            # Create a point cloud from the RGBD image
            pcd = self.convert_to_pointcloud(rgbd_image)

            # Segment the planes in the point cloud
            segments = self.segment_planes(pcd)

            # Check if the publishers are valid before publishing messages
            if self.planes_publisher.get_num_connections() == 0:
                rospy.loginfo("No subscribers to the '/planes' topic.")

            else:

                # Publish each segmented plane as a ROS message
                for i, segment in enumerate(segments):
                    header = Header()
                    header.stamp = rospy.Time.now()
                    header.frame_id = "base_link"  # Set the appropriate frame ID
                    planes_msg = pc2.create_cloud_xyz32(header, segment.points)  # Use o3d.io.create_point_cloud2_msg
                    self.planes_publisher.publish(planes_msg)

            if self.planes_visual_publisher.get_num_connections() == 0:
                rospy.loginfo("No subscribers to /planes_visual topic")

            else:
                # Publish the segmented planes for visualization
                self.publish_planes_visual(segments)

    def depth_callback(self, depth_msg):

        # Convert the ROS depth Image to np array
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # Convert the depth Image to float32 and normalize it in the range[0,1]
        depth_image = depth_image.astype(np.float32)
        depth_image /= 65535.0  # Assuming the depth values are in the range [0, 65535]

        # Convert the depth image to Open3D image
        self.depth_image = o3d.geometry.Image(np.array(depth_image))

    def convert_to_pointcloud(self, rgbd_image):
        # Convert RGBD images to numpy arrays
        # color_np = np.array(rgbd_image.color)
        # depth_np = np.array(rgbd_image.depth)

        # Get color image dimensions
        color_data = np.asarray(rgbd_image.color)
        color_height, color_width = color_data.shape[:2]

        # Create a pinhole camera intrinsic
        intrinsics = o3d.camera.PinholeCameraIntrinsic(color_width, color_height, self.fx, self.fy, self.cx, self.cy)

        # Create an identity extrinsic matrix
        extrinsics = np.eye(4)

        # Create a point cloud from the RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsics, extrinsics, project_valid_depth_only=True)

        return pcd

    def segment_planes(self, point_cloud):
        """
        Segment planes in a point cloud using Open3D.

        Args:
            point_cloud (open3d.geometry.PointCloud): Input point cloud.

        Returns:
            List[open3d.geometry.PointCloud]: List of segmented planes.
        """
        # Convert the point cloud to numpy array
        points = np.asarray(point_cloud.points)

        # Create an Open3D point cloud from the numpy array
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Check if the point cloud has enough points for segmentation
        if len(points) < 3:
            rospy.logwarn("Insufficient points in the point cloud for plane segmentation.")
            return []

        # Perform plane segmentation
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)

        # Separate the plane points and the outlier points
        plane_points = pcd.select_by_index(inliers)
        outlier_points = pcd.select_by_index(inliers, invert=True)

        # Recursively segment additional planes from the outlier points
        segmented_planes = []
        if len(outlier_points.points) > 0:
            segmented_planes = self.segment_planes(outlier_points)

        # Append the current plane to the segmented planes list
        segmented_planes.append(plane_points)

        return segmented_planes

    def publish_planes_visual(self, segmented_planes):
        """
        Publish segmented planes as visualization using MarkerArray.

        Args:
            segmented_planes (List[open3d.geometry.PointCloud]): List of segmented planes.
        """
        # Check if the publisher is valid
        if self.planes_visual_publisher.get_num_connections() == 0:
            return

        # Create a MarkerArray message for visualization
        marker_array = MarkerArray()

        # Define a list of colors to be used for each segmented plane
        colors = [[1.0, 0.0, 0.0],  # Red
                  [0.0, 1.0, 0.0],  # Green
                  [0.0, 0.0, 1.0],  # Blue
                  [1.0, 1.0, 0.0],  # Yellow
                  [1.0, 0.0, 1.0],  # Magenta
                  [0.0, 1.0, 1.0],  # Cyan
                  # Add more colors here if needed
                  ]

        for i, plane_points in enumerate(segmented_planes):
            # Create a Marker message for each segmented plane
            marker = Marker()
            marker.header.frame_id = "base_link"  # Set the appropriate frame ID
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0

            # Use a unique color for each segmented plane
            color = colors[i % len(colors)]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]

            # Convert the plane points to a PointCloud2 message
            points = np.asarray(plane_points.points)
            points_msg = pc2.create_cloud_xyz32(marker.header, points)

            # Parse the PointCloud2 data and extract the points
            for p in pc2.read_points(points_msg, field_names=("x", "y", "z"), skip_nans=True):
                point = Point()
                point.x = p[0]
                point.y = p[1]
                point.z = p[2]
                marker.points.append(point)

            # Append the Marker message to the MarkerArray message
            marker_array.markers.append(marker)

        # Publish the MarkerArray message for visualization
        self.planes_visual_publisher.publish(marker_array)

    def run(self):
        # Keep the node running until it is shutdown
        rospy.spin()


if __name__ == "__main__":
    node = PlaneSegmentationNode()
    node.run()

