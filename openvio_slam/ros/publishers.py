"""
ros/publishers.py
=================
All ROS 2 publishing logic for OpenVIO-SLAM, separated from the main node.

Why a Separate Module?
    vio_node.py orchestrates the pipeline (subscribe → process → publish).
    If publishing logic lives there too, the node becomes a 600-line monolith
    that's hard to test or modify. Here we separate:

        vio_node.py    — pipeline orchestration, subscription callbacks, threading
        publishers.py  — message construction and topic publishing

    This enables:
        - Unit tests that construct messages without a running ROS node
        - Easy addition of new topics without touching the main node
        - Clean separation of ROS plumbing from SLAM logic

Published Topics:
    /odom                   nav_msgs/Odometry          — VIO pose + twist estimate
    /tf                     tf2                        — odom→base_link, map→odom
    /openvio/map_points     sensor_msgs/PointCloud2    — sparse 3D landmark cloud
    /openvio/keyframes      visualization_msgs/Marker  — camera trajectory (line strip)
    /openvio/covariance     visualization_msgs/Marker  — position uncertainty ellipsoid
    /openvio/loop_closures  visualization_msgs/Marker  — loop closure edges (red lines)
    /openvio/features       sensor_msgs/Image          — debug image with tracked features

Services Handled:
    /save_map   std_srvs/Trigger   — delegated from node, publishes status
    /load_map   std_srvs/SetString — delegated from node, publishes status
"""

import struct
import numpy as np
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from sensor_msgs.msg         import Image, PointCloud2, PointField
from nav_msgs.msg            import Odometry, Path
from geometry_msgs.msg       import (TransformStamped, PoseStamped,
                                     PoseWithCovarianceStamped, Point)
from visualization_msgs.msg  import Marker, MarkerArray
from std_msgs.msg            import ColorRGBA, Header
from builtin_interfaces.msg  import Time
from tf2_ros                 import TransformBroadcaster

import cv2


# ─────────────────────────────────────────────────────────
# Header + timestamp helpers
# ─────────────────────────────────────────────────────────

def make_header(stamp, frame_id: str) -> Header:
    h = Header()
    h.stamp    = stamp
    h.frame_id = frame_id
    return h


def make_color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def make_point(x: float, y: float, z: float) -> Point:
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


# ─────────────────────────────────────────────────────────
# Odometry publisher
# ─────────────────────────────────────────────────────────

class OdometryPublisher:
    """
    Publishes nav_msgs/Odometry and broadcasts odom→base_link TF.

    Includes:
        - Full 6-DOF pose (position + orientation quaternion)
        - Linear + angular velocity
        - Pose covariance from the MSCKF position/orientation subblock
    """

    def __init__(self, node: Node):
        self._pub    = node.create_publisher(Odometry, '/odom', 10)
        self._tf_br  = TransformBroadcaster(node)
        self._path_pub = node.create_publisher(Path, '/openvio/path', 10)
        self._path   = Path()
        self._max_path_len = 500   # drop oldest after this many waypoints

    def publish(
        self,
        stamp,
        q_GI: np.ndarray,
        p_GI: np.ndarray,
        v_GI: np.ndarray,
        P:    np.ndarray,
        child_frame: str = 'base_link',
        parent_frame: str = 'odom',
    ):
        """
        Args:
            stamp:       ROS2 time stamp
            q_GI:        (4,) [qw, qx, qy, qz] world←IMU quaternion
            p_GI:        (3,) position in world frame [m]
            v_GI:        (3,) velocity in world frame [m/s]
            P:           (≥15, ≥15) MSCKF state covariance
            child_frame: child frame ID (default 'base_link')
            parent_frame: parent frame ID (default 'odom')
        """
        qw, qx, qy, qz = float(q_GI[0]), float(q_GI[1]), float(q_GI[2]), float(q_GI[3])
        px, py, pz      = float(p_GI[0]), float(p_GI[1]), float(p_GI[2])
        vx, vy, vz      = float(v_GI[0]), float(v_GI[1]), float(v_GI[2])

        # ── Odometry message ─────────────────────────────
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = parent_frame
        odom.child_frame_id  = child_frame

        odom.pose.pose.position.x    = px
        odom.pose.pose.position.y    = py
        odom.pose.pose.position.z    = pz
        odom.pose.pose.orientation.w = qw
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.twist.twist.linear.x    = vx
        odom.twist.twist.linear.y    = vy
        odom.twist.twist.linear.z    = vz

        # Pose covariance (6×6 from MSCKF P — rotation [0:3] + position [3:6])
        # ROS convention: [x, y, z, rot_x, rot_y, rot_z] (translation first)
        if P is not None and P.shape[0] >= 6:
            P_pos = P[3:6, 3:6]    # position covariance
            P_rot = P[0:3, 0:3]    # orientation error covariance
            cov = np.zeros(36)
            cov[0], cov[7],  cov[14] = P_pos[0,0], P_pos[1,1], P_pos[2,2]
            cov[21], cov[28], cov[35] = P_rot[0,0], P_rot[1,1], P_rot[2,2]
            odom.pose.covariance = cov.tolist()

        self._pub.publish(odom)

        # ── TF broadcast ─────────────────────────────────
        tf_msg = TransformStamped()
        tf_msg.header.stamp    = stamp
        tf_msg.header.frame_id = parent_frame
        tf_msg.child_frame_id  = child_frame
        tf_msg.transform.translation.x = px
        tf_msg.transform.translation.y = py
        tf_msg.transform.translation.z = pz
        tf_msg.transform.rotation.w    = qw
        tf_msg.transform.rotation.x    = qx
        tf_msg.transform.rotation.y    = qy
        tf_msg.transform.rotation.z    = qz
        self._tf_br.sendTransform(tf_msg)

        # ── Path ─────────────────────────────────────────
        ps = PoseStamped()
        ps.header = odom.header
        ps.pose   = odom.pose.pose
        self._path.header = odom.header
        self._path.poses.append(ps)
        if len(self._path.poses) > self._max_path_len:
            self._path.poses = self._path.poses[-self._max_path_len:]
        self._path_pub.publish(self._path)


# ─────────────────────────────────────────────────────────
# Map points (PointCloud2) publisher
# ─────────────────────────────────────────────────────────

class MapPointsPublisher:
    """
    Publishes the sparse 3D landmark map as sensor_msgs/PointCloud2.
    Color-encodes point quality (found_ratio): green=good, red=poor.
    """

    def __init__(self, node: Node, max_points: int = 2000):
        self._pub        = node.create_publisher(PointCloud2, '/openvio/map_points', 5)
        self._max_points = max_points

    def publish(self, stamp, map_points: list, frame_id: str = 'odom'):
        """
        Args:
            stamp:       ROS2 timestamp
            map_points:  List of MapPoint objects (with .position and .found_ratio)
            frame_id:    Frame the points are expressed in (default 'odom')
        """
        if not map_points:
            return

        # Downsample if needed
        pts = map_points
        if len(pts) > self._max_points:
            # Keep highest found_ratio points
            pts = sorted(pts, key=lambda p: p.found_ratio, reverse=True)[:self._max_points]

        # PointCloud2 with XYZ + RGB
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        data = bytearray()
        for mp in pts:
            x, y, z = float(mp.position[0]), float(mp.position[1]), float(mp.position[2])
            # Color: green (good) → red (poor) based on found_ratio
            ratio = float(mp.found_ratio)
            r_c   = int((1.0 - ratio) * 255)
            g_c   = int(ratio * 255)
            rgb_int = (r_c << 16) | (g_c << 8) | 0
            rgb_f   = struct.unpack('f', struct.pack('I', rgb_int))[0]
            data += struct.pack('ffff', x, y, z, rgb_f)

        cloud = PointCloud2()
        cloud.header.stamp    = stamp
        cloud.header.frame_id = frame_id
        cloud.height     = 1
        cloud.width      = len(pts)
        cloud.fields     = fields
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step   = 16 * len(pts)
        cloud.is_dense   = True
        cloud.data       = bytes(data)
        self._pub.publish(cloud)


# ─────────────────────────────────────────────────────────
# Camera trajectory (keyframe poses) publisher
# ─────────────────────────────────────────────────────────

class KeyframeTrajectoryPublisher:
    """
    Publishes keyframe camera poses as a LINE_STRIP marker (thin blue line)
    and individual ARROW markers for each keyframe orientation.
    """

    def __init__(self, node: Node):
        self._pub = node.create_publisher(MarkerArray, '/openvio/keyframes', 5)

    def publish(self, stamp, keyframes: list, frame_id: str = 'odom'):
        """
        Args:
            keyframes: List of Keyframe objects (with .p_GC and .q_GC)
        """
        if not keyframes:
            return

        markers = MarkerArray()

        # ── Line strip connecting all keyframe positions ──
        line_marker = Marker()
        line_marker.header.stamp    = stamp
        line_marker.header.frame_id = frame_id
        line_marker.ns     = 'kf_trajectory'
        line_marker.id     = 0
        line_marker.type   = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.02    # line width [m]
        line_marker.color   = make_color(0.2, 0.6, 1.0, 0.8)   # light blue

        for kf in keyframes:
            p = kf.p_GC
            line_marker.points.append(make_point(p[0], p[1], p[2]))

        markers.markers.append(line_marker)

        # ── Arrow for most recent keyframe (camera viewing direction) ──
        if keyframes:
            kf = keyframes[-1]
            from ..utils.quaternion import quat_to_rot
            R  = quat_to_rot(kf.q_GC)
            # Camera z-axis (viewing direction) in world frame
            view_dir = R[:, 2]
            p = kf.p_GC

            arrow = Marker()
            arrow.header.stamp    = stamp
            arrow.header.frame_id = frame_id
            arrow.ns     = 'kf_current'
            arrow.id     = 1
            arrow.type   = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x = 0.03; arrow.scale.y = 0.06; arrow.scale.z = 0.0
            arrow.color   = make_color(1.0, 0.5, 0.0, 1.0)   # orange
            arrow.points.append(make_point(p[0], p[1], p[2]))
            tip = p + view_dir * 0.3
            arrow.points.append(make_point(tip[0], tip[1], tip[2]))
            markers.markers.append(arrow)

        self._pub.publish(markers)


# ─────────────────────────────────────────────────────────
# Loop closure edges publisher
# ─────────────────────────────────────────────────────────

class LoopClosurePublisher:
    """
    Publishes loop closure edges as red LINE_LIST markers in RViz.
    Each accepted loop closure adds a line between the two matched keyframes.
    """

    def __init__(self, node: Node):
        self._pub   = node.create_publisher(MarkerArray, '/openvio/loop_closures', 5)
        self._edges: List[Tuple[np.ndarray, np.ndarray]] = []   # (p_query, p_match) pairs

    def add_loop_closure(self, p_query: np.ndarray, p_match: np.ndarray):
        """Register a new loop closure edge to be visualized."""
        self._edges.append((p_query.copy(), p_match.copy()))

    def publish(self, stamp, frame_id: str = 'odom'):
        if not self._edges:
            return

        marker = Marker()
        marker.header.stamp    = stamp
        marker.header.frame_id = frame_id
        marker.ns     = 'loop_closures'
        marker.id     = 0
        marker.type   = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.04           # line width
        marker.color   = make_color(1.0, 0.0, 0.0, 0.9)   # red

        for p_q, p_m in self._edges:
            marker.points.append(make_point(p_q[0], p_q[1], p_q[2]))
            marker.points.append(make_point(p_m[0], p_m[1], p_m[2]))

        arr = MarkerArray()
        arr.markers.append(marker)
        self._pub.publish(arr)


# ─────────────────────────────────────────────────────────
# Uncertainty ellipsoid publisher
# ─────────────────────────────────────────────────────────

class UncertaintyPublisher:
    """
    Publishes a SPHERE marker sized to the position uncertainty (1-sigma ellipsoid).
    The sphere radius = sqrt(max eigenvalue of position covariance block).
    """

    def __init__(self, node: Node):
        self._pub = node.create_publisher(Marker, '/openvio/uncertainty', 5)

    def publish(self, stamp, p_GI: np.ndarray, P: np.ndarray, frame_id: str = 'odom'):
        if P is None or P.shape[0] < 6:
            return

        P_pos = P[3:6, 3:6]
        try:
            max_eig = float(np.linalg.eigvalsh(P_pos).max())
        except np.linalg.LinAlgError:
            return

        radius = float(np.sqrt(max(max_eig, 1e-6)))
        radius = min(radius, 5.0)   # clamp to avoid huge sphere before convergence

        m = Marker()
        m.header.stamp    = stamp
        m.header.frame_id = frame_id
        m.ns     = 'uncertainty'
        m.id     = 0
        m.type   = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(p_GI[0])
        m.pose.position.y = float(p_GI[1])
        m.pose.position.z = float(p_GI[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = radius * 2.0   # diameter
        # Color: blue (confident) → yellow (uncertain)
        confidence = max(0.0, min(1.0, 1.0 - radius / 1.0))
        m.color = make_color(1.0 - confidence, confidence * 0.8, confidence, 0.35)
        self._pub.publish(m)


# ─────────────────────────────────────────────────────────
# Debug image publisher (tracked features overlay)
# ─────────────────────────────────────────────────────────

class DebugImagePublisher:
    """
    Publishes a debug image with tracked feature points overlaid.

    Green circles: actively tracked features
    Red dots:      features that were just lost
    Yellow text:   feature count and tracking stats
    """

    def __init__(self, node: Node):
        self._pub = node.create_publisher(Image, '/openvio/debug_image', 5)

    def publish(
        self,
        stamp,
        gray_image:      np.ndarray,
        active_pts:      Optional[np.ndarray],
        lost_pts:        Optional[np.ndarray] = None,
        status_text:     str = '',
    ):
        """
        Args:
            gray_image:  (H, W) uint8 grayscale
            active_pts:  (N, 2) float32 tracked feature positions [u, v]
            lost_pts:    (M, 2) float32 features lost this frame (optional)
            status_text: Status string to overlay (e.g. "INITIALIZED | 247 feats")
        """
        # Convert to BGR for colored overlays
        bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        if active_pts is not None and len(active_pts) > 0:
            for pt in active_pts:
                if np.all(np.isfinite(pt)):
                    cv2.circle(bgr, (int(pt[0]), int(pt[1])), 4,
                               (0, 220, 0), 1, cv2.LINE_AA)   # green

        if lost_pts is not None and len(lost_pts) > 0:
            for pt in lost_pts:
                if np.all(np.isfinite(pt)):
                    cv2.circle(bgr, (int(pt[0]), int(pt[1])), 3,
                               (0, 0, 200), -1, cv2.LINE_AA)   # red filled

        if status_text:
            cv2.putText(bgr, status_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 50), 1, cv2.LINE_AA)

        n_feats = len(active_pts) if active_pts is not None else 0
        cv2.putText(bgr, f"Tracking: {n_feats}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Build sensor_msgs/Image
        img_msg = Image()
        img_msg.header.stamp    = stamp
        img_msg.header.frame_id = 'camera'
        img_msg.height    = bgr.shape[0]
        img_msg.width     = bgr.shape[1]
        img_msg.encoding  = 'bgr8'
        img_msg.is_bigendian = False
        img_msg.step      = bgr.shape[1] * 3
        img_msg.data      = bgr.tobytes()
        self._pub.publish(img_msg)


# ─────────────────────────────────────────────────────────
# Aggregate publisher (single object to pass to the node)
# ─────────────────────────────────────────────────────────

class SLAMPublishers:
    """
    Convenience wrapper that holds all publishers and provides a single
    publish_all() interface for the main node.

    Usage in vio_node.py:
        self._publishers = SLAMPublishers(self)
        ...
        self._publishers.publish_all(stamp, state, map_manager, ...)
    """

    def __init__(self, node: Node, max_map_points: int = 2000):
        self.odom          = OdometryPublisher(node)
        self.map_points    = MapPointsPublisher(node, max_map_points)
        self.keyframes     = KeyframeTrajectoryPublisher(node)
        self.loop_closures = LoopClosurePublisher(node)
        self.uncertainty   = UncertaintyPublisher(node)
        self.debug_image   = DebugImagePublisher(node)

        self._frame_count  = 0

    def publish_odometry(self, stamp, state):
        """Publish /odom and TF at every frame (high frequency)."""
        self.odom.publish(
            stamp,
            q_GI=state.q_GI,
            p_GI=state.p_GI,
            v_GI=state.v_GI,
            P=state.P,
        )
        self.uncertainty.publish(stamp, state.p_GI, state.P)

    def publish_map(self, stamp, map_manager, state):
        """Publish map-related topics at lower frequency."""
        self._frame_count += 1

        # Map points: 1 Hz
        if self._frame_count % 10 == 0:
            mp_list = map_manager.get_all_map_points()
            self.map_points.publish(stamp, mp_list)

        # Keyframe trajectory: 2 Hz
        if self._frame_count % 5 == 0:
            kf_list = list(map_manager._keyframes.values())
            self.keyframes.publish(stamp, kf_list)

        # Loop closures: 0.5 Hz (only if there are any)
        if self._frame_count % 20 == 0 and map_manager.n_loop_closures > 0:
            self.loop_closures.publish(stamp)

    def publish_debug(self, stamp, gray, active_pts, status: str = ''):
        """Publish debug image with overlay."""
        self.debug_image.publish(stamp, gray, active_pts, status_text=status)

    def register_loop_closure(self, p_query: np.ndarray, p_match: np.ndarray):
        """Register a loop closure for visualization."""
        self.loop_closures.add_loop_closure(p_query, p_match)
